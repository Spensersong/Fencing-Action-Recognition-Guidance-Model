"""Use the opening guard as a reference and check which pose frames can be used."""

from __future__ import annotations

import numpy as np

from .geometry import (
    CORE,
    SIDES,
    bone_lengths,
    measurements,
    other,
    pixels,
    unit,
    view_for_axis,
    visible,
)
from .schema import Calibration, FeatureFrame


def calibrate(frames, config):
    if not frames:
        return Calibration(False, "No frames were supplied.")
    if not config.camera_level:
        return Calibration(False, "A level camera is required for the configured reference frame.")
    end = frames[0].t + config.calibration_seconds
    candidates = [f for f in frames if f.t <= end and visible(f, CORE, config.visibility_threshold)]
    if len(candidates) < 5 or candidates[-1].t - candidates[0].t < 0.45:
        return Calibration(
            False, "Insufficient visible calibration data. Hold en garde for the first 1.2 seconds."
        )
    if (
        len(candidates) / max(1, sum(f.t <= end for f in frames)) < config.min_coverage
        or max(b.t - a.t for a, b in zip(candidates, candidates[1:], strict=False))
        > config.max_gap_seconds + 1e-6
    ):
        return Calibration(
            False, "Calibration is interrupted by missing or unreliable observations."
        )
    front, rear = SIDES[config.front_side], SIDES[other(config.front_side)]
    vectors = [f.world[front["ankle"], :3] - f.world[rear["ankle"], :3] for f in candidates]
    vector = np.median(vectors, axis=0)
    vector[1] = 0
    axis = unit(vector)
    if axis is None:
        return Calibration(False, "The front-to-rear foot direction is not observable.")
    lengths = np.median([bone_lengths(f.world) for f in candidates], axis=0)
    if not np.isfinite(lengths).all() or np.min(lengths) < 0.08 or np.max(lengths) > 1.3:
        return Calibration(False, "Implausible limb geometry during calibration.")
    leg_length = float(np.sum(lengths[:4]) / 2)
    pixel_vectors = [pixels(f)[front["ankle"]] - pixels(f)[rear["ankle"]] for f in candidates]
    screen_axis = unit(np.median(pixel_vectors, axis=0))
    if screen_axis is None:
        screen_axis = np.array([1.0, 0.0])
    screen_scale = float(
        np.median(
            [
                np.linalg.norm((pixels(f)[11] + pixels(f)[12] - pixels(f)[23] - pixels(f)[24]) / 2)
                for f in candidates
            ]
        )
    )
    if screen_scale < 20:
        return Calibration(False, "The person is too small in the image; move the camera closer.")
    values = [
        measurements(f, config, axis, leg_length, screen_axis, screen_scale) for f in candidates
    ]
    baseline = {}
    for key in values[0]:
        available = [v[key] for v in values if v[key] is not None]
        if available:
            baseline[key] = float(np.median(available))
    if not (
        90 <= baseline["front_knee"] <= 165
        and 90 <= baseline["rear_knee"] <= 165
        and 0.25 <= baseline["stance"] <= 1.45
    ):
        return Calibration(
            False,
            "Initial posture is outside the supported en-garde calibration range. Bend both knees and separate the feet.",
        )
    if (
        np.std([v["stance"] for v in values]) > 0.07
        or np.std([v["hip_screen"] for v in values]) > 0.12
    ):
        return Calibration(
            False, "Calibration was moving. Keep the camera and en-garde posture still initially."
        )
    view, projection = view_for_axis(axis)
    return Calibration(
        True,
        "Calibration accepted; world coordinates are monocular estimates, not motion-capture measurements.",
        end=end,
        axis=axis.tolist(),
        screen_axis=screen_axis.tolist(),
        leg_length=leg_length,
        screen_scale=screen_scale,
        baseline=baseline,
        bone_lengths=lengths.tolist(),
        view=view,
        projection=projection,
        movement_observable=projection >= 0.45 and config.camera_static and config.camera_level,
    )


def extract_features(frames, calibration, config):
    if any(not np.isfinite(f.t) for f in frames):
        raise ValueError("Timestamps must be finite.")
    if any(b.t <= a.t for a, b in zip(frames, frames[1:], strict=False)):
        raise ValueError("Timestamps must be strictly increasing.")
    if not calibration.valid:
        return [FeatureFrame(f.t, False, reasons=[calibration.reason]) for f in frames]
    output = []
    previous = None
    axis = np.asarray(calibration.axis)
    for frame in frames:
        reasons = []
        if frame.people > 1:
            reasons.append("Multiple people detected; subject identity is ambiguous.")
        elif not visible(frame, CORE, config.visibility_threshold):
            reasons.append(
                "Required body landmarks are missing, cropped, or below the visibility threshold."
            )
        if not reasons:
            lengths = bone_lengths(frame.world)
            if np.any(abs(lengths / calibration.bone_lengths - 1) > 0.35):
                reasons.append(
                    "Estimated limb lengths changed excessively; tracking is unreliable."
                )
        if not config.camera_level:
            reasons.append("A level camera is required for the configured reference frame.")
        if reasons:
            output.append(FeatureFrame(frame.t, False, reasons=reasons))
            previous = None
            continue
        values = measurements(
            frame,
            config,
            axis,
            calibration.leg_length,
            calibration.screen_axis,
            calibration.screen_scale,
        )
        front, rear = SIDES[config.front_side], SIDES[other(config.front_side)]
        current_axis = frame.world[front["ankle"], :3] - frame.world[rear["ankle"], :3]
        current_axis[1] = 0
        current_axis = unit(current_axis)
        view, _ = view_for_axis(current_axis) if current_axis is not None else ("unknown", 0)
        if current_axis is None or np.dot(current_axis, axis) < np.cos(np.radians(32)):
            reasons.append("Movement axis changed; recalibration is required.")
        if previous is not None:
            dt = frame.t - previous.t
            if dt <= config.max_gap_seconds and abs(
                values["hip_screen"] - previous.values["hip_screen"]
            ) > max(1.2, 12 * dt):
                reasons.append("Abrupt image displacement; camera cut or tracking jump.")
        quality = float(np.min(frame.image[CORE, 3:5]))
        item = FeatureFrame(frame.t, not reasons, values, reasons, view, quality)
        output.append(item)
        previous = item if item.valid else None
    return output


def continuous_runs(features, max_gap=0.20):
    """Start a new run after a missing frame or a gap in the recording."""
    run = []
    for f in features:
        if not f.valid or (run and f.t - run[-1].t > max_gap + 1e-6):
            if run:
                yield run
            run = []
        if f.valid:
            run.append(f)
    if run:
        yield run


def smooth_run(run, radius_seconds=0.065):
    """Take the median over nearby frames within the same run."""
    times = np.array([f.t for f in run])
    result = []
    for f in run:
        neighbours = run[
            np.searchsorted(times, f.t - radius_seconds) : np.searchsorted(
                times, f.t + radius_seconds, side="right"
            )
        ]
        values = {}
        for key, value in f.values.items():
            items = [n.values[key] for n in neighbours if n.values.get(key) is not None]
            values[key] = float(np.median(items)) if value is not None and items else None
        result.append(FeatureFrame(f.t, True, values, [], f.view, f.quality))
    return result
