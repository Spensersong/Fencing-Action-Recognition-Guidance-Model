"""Measurement -> explicit reference range -> fixed coaching cue.

Missing values never become zero, ideal angles, or invented observations.
An overall /100 score requires ALL weighted metrics to be supported.
"""

from __future__ import annotations

import json
from importlib.resources import files

import numpy as np

from .schema import DIMENSIONS, Action, ActionResult, MetricResult


def load_rubric():
    return json.loads(files("fencing_coach").joinpath("rubric.json").read_text(encoding="utf-8"))


def range_score(value, target, limits):
    low, high = target
    outer_low, outer_high = limits
    if low <= value <= high:
        return 100.0
    if value < low:
        return float(np.clip(100 * (value - outer_low) / max(low - outer_low, 1e-8), 0, 100))
    return float(np.clip(100 * (outer_high - value) / max(outer_high - high, 1e-8), 0, 100))


def onset(times, signal, threshold, sustain=0.06):
    """First sustained crossing. A single noisy frame is not an event."""
    for i, value in enumerate(signal):
        if not np.isfinite(value) or value < threshold:
            continue
        j = int(np.searchsorted(times, times[i] + sustain))
        if j < len(times) and np.all(np.asarray(signal)[i : j + 1] >= threshold):
            return float(times[i])
    return None


def measure_action(action, features, calibration, config):
    rows = [f for f in features if action.start - 1e-5 <= f.t <= action.end + 1e-5]
    good = [f for f in rows if f.valid]
    if len(good) < 3 or len(good) / max(1, len(rows)) < config.min_coverage:
        return {}, "Insufficient reliable frames in this action."
    t = np.array([f.t for f in good])
    if np.max(np.diff(t)) > config.max_gap_seconds + 1e-6:
        return {}, "An observation gap interrupts this action."
    data = {}

    def series(key):
        return np.array(
            [f.values.get(key) if f.values.get(key) is not None else np.nan for f in good]
        )

    def save(key, value, at, reason=""):
        if value is not None and np.isfinite(value):
            data[key] = (float(value), float(at), reason)

    peak_mask = abs(t - action.peak) <= max(0.065, float(np.median(np.diff(t))))
    end_mask = t >= t[-1] - 0.14
    for key in ("front_knee", "rear_knee", "elbow", "trunk", "knee_offset"):
        values = series(key)
        mask = np.isfinite(values) if action.kind == "en_garde" else peak_mask & np.isfinite(values)
        if np.sum(mask) >= 2:
            save(key, np.median(values[mask]), action.peak)
        if (
            key in {"front_knee", "rear_knee", "trunk"}
            and np.sum(end_mask & np.isfinite(values)) >= 2
        ):
            save(key + "_end", np.median(values[end_mask & np.isfinite(values)]), t[-1])
    stance = series("stance")
    save("stance_error", abs(np.median(stance[end_mask]) - calibration.baseline["stance"]), t[-1])
    save("stance_min", np.min(stance), t[np.argmin(stance)])
    trunk = series("trunk")
    if np.isfinite(trunk).all():
        save("trunk_variation", np.std(trunk), t[-1])
    save("duration", action.end - action.start, action.end)

    # Image-space measures are withheld when translation/projection is ambiguous.
    if calibration.movement_observable and all(f.view in {"side", "oblique"} for f in good):
        front, rear = series("front_screen"), series("rear_screen")
        hip_y = series("hip_height")
        save("hip_range", np.ptp(hip_y), t[np.argmax(hip_y)])
        save("foot_drift", max(np.ptp(front), np.ptp(rear)), t[-1])
        save("final_drift", np.ptp(series("hip_screen")[end_mask]), t[-1])
        sign = -1 if action.kind == "retreat" else 1
        front_onset = onset(t, sign * (front - front[0]), 0.04)
        rear_onset = onset(t, sign * (rear - rear[0]), 0.04)
        resolution = max(0.07, 2 * float(np.median(np.diff(t))))
        if front_onset is not None and rear_onset is not None:
            delay = (
                rear_onset - front_onset if action.kind != "retreat" else front_onset - rear_onset
            )
            if abs(delay) > resolution:
                save("foot_order", delay, max(front_onset, rear_onset))
        elbow, reach = series("elbow"), series("reach")
        arm_onset = onset(t, elbow - calibration.baseline.get("elbow", np.nan), 8)
        if (
            arm_onset is not None
            and front_onset is not None
            and abs(front_onset - arm_onset) > resolution
        ):
            save("arm_lead", front_onset - arm_onset, max(arm_onset, front_onset))
    else:
        arm_onset = onset(t, series("elbow") - calibration.baseline.get("elbow", np.nan), 8)

    if arm_onset is not None and action.peak > arm_onset:
        save("attack_duration", action.peak - arm_onset, action.peak)
    elif action.kind in {"lunge", "advance_lunge"}:
        stance_onset = onset(t, stance - calibration.baseline["stance"], 0.045)
        if stance_onset is not None and action.peak > stance_onset:
            save("attack_duration", action.peak - stance_onset, action.peak)
    reach = series("reach")
    path = reach[
        (t <= action.peak + 1e-5) & (t >= (arm_onset if arm_onset is not None else action.start))
    ]
    if len(path) >= 4 and np.isfinite(path).all() and np.ptp(path) > 0.12:
        save("reach_efficiency", max(1, np.sum(abs(np.diff(path))) / np.ptp(path)), action.peak)
    return data, ""


def score_action(action, features, calibration, config, rubric=None):
    rubric = rubric or load_rubric()
    if action.kind == "unknown":
        return ActionResult(
            action,
            [],
            dict.fromkeys(DIMENSIONS),
            0.0,
            None,
            "unclassified",
            [action.evidence.get("reason", "Unsupported movement.")],
        )
    if action.kind == "advance_lunge":
        start, end = action.evidence["advance_interval"]
        step = score_action(
            Action("advance", start, end, end, action.complete),
            features,
            calibration,
            config,
            rubric,
        )
        lunge = score_action(
            Action(
                "lunge", action.evidence["lunge_start"], action.end, action.peak, action.complete
            ),
            features,
            calibration,
            config,
            rubric,
        )
        metrics = []
        for prefix, result in (("Advance", step), ("Lunge", lunge)):
            for m in result.metrics:
                metrics.append(
                    MetricResult(
                        prefix.lower() + "." + m.key,
                        prefix + ": " + m.label,
                        m.dimension,
                        m.weight / 2,
                        m.value,
                        m.unit,
                        m.target,
                        m.score,
                        m.observed_at,
                        m.reason,
                        m.cue,
                        m.source,
                    )
                )
        return summarize(
            action,
            metrics,
            config,
            ["Compound score gives equal weight to the advance and lunge components."],
        )
    profile = "step" if action.kind in {"advance", "retreat"} else action.kind
    data, failure = measure_action(action, features, calibration, config)
    metrics = []
    for key, label, dimension, weight, unit, target, limits, cue, source in rubric["profiles"][
        profile
    ]:
        value, at, _ = data.get(key, (None, None, ""))
        reason = failure
        if value is None and not reason:
            if (
                key in {"hip_range", "foot_drift", "final_drift", "foot_order", "arm_lead"}
                and not calibration.movement_observable
            ):
                reason = "Requires a fixed, level side/oblique view with observable axial movement."
            elif key in {"arm_lead", "foot_order"}:
                reason = "Separate sustained onsets were not resolved at this frame rate; ordering is withheld."
            else:
                reason = "Required landmarks or sustained motion were not sufficiently observed."
        # Classical arm-extension targets are not imposed on sabre technique.
        if config.weapon == "sabre" and key in {"arm_lead", "elbow", "reach_efficiency"}:
            value, at, reason = (
                None,
                None,
                "This classical thrust/extension rule is not graded for sabre.",
            )
        if value is not None and not target[0] <= value <= target[1]:
            direction = "below" if value < target[0] else "above"
            cue = rubric.get("directional_cues", {}).get(f"{profile}.{key}.{direction}", cue)
        metrics.append(
            MetricResult(
                key,
                label,
                dimension,
                weight,
                value,
                unit,
                target,
                range_score(value, target, limits) if value is not None else None,
                at,
                reason,
                cue,
                source,
            )
        )
    return summarize(action, metrics, config, [])


def summarize(action, metrics, config, notes):
    observed = [m for m in metrics if m.score is not None]
    coverage = sum(m.weight for m in observed) / 100
    dimensions = {}
    for key, (_, weight) in DIMENSIONS.items():
        items = [m for m in metrics if m.dimension == key]
        dimensions[key] = (
            round(sum(m.weight * m.score / 100 for m in items), 2)
            if items and all(m.score is not None for m in items)
            else None
        )
        if abs(sum(m.weight for m in items) - weight) > 1e-6:
            raise ValueError(f"Rubric weights do not match the {key} contract.")
    if not action.complete:
        notes.append(
            "The action was cut off or its finish was not observed; overall score withheld."
        )
    total = (
        round(sum(m.weight * m.score / 100 for m in observed), 1)
        if coverage >= 1 - 1e-8 and action.complete
        else None
    )
    status = "scored" if total is not None else "partial" if observed else "not_assessable"
    return ActionResult(action, metrics, dimensions, round(coverage, 4), total, status, notes)
