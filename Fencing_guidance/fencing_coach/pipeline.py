"""Read a saved clip or the local webcam and write the report files."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

from .analysis import analyze_frames
from .report import write_pdf
from .schema import ACTION_NAMES, AnalysisConfig, PoseFrame
from .vision import DEFAULT_POSE_MODEL, PoseExtractor, SwordSegmenter, draw_overlay, sha256


def save_outputs(result, features, frames, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = write_pdf(result, output_dir / "coaching_report.pdf", frames)
    payload = result.to_dict()
    (output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "landmarks.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "simulated": result.simulated,
                "source_name": result.source_name,
                "frames": [f.to_dict() for f in frames],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    keys = sorted({key for f in features for key in f.values})
    with open(output_dir / "measurements.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["time_seconds", "valid", "view", "quality", "exclusion_reason"] + keys,
        )
        writer.writeheader()
        for feature in features:
            writer.writerow(
                {
                    "time_seconds": feature.t,
                    "valid": feature.valid,
                    "view": feature.view,
                    "quality": feature.quality,
                    "exclusion_reason": "; ".join(feature.reasons),
                    **feature.values,
                }
            )
    return {
        "pdf": pdf,
        "json": output_dir / "analysis.json",
        "landmarks": output_dir / "landmarks.json",
        "csv": output_dir / "measurements.csv",
    }


def extract_video(path, config, pose_model=DEFAULT_POSE_MODEL, sword_model=None, progress=None):
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video. Convert it to an H.264 MP4 file.")
    frames, warnings = [], []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps < 1 or fps > 240:
            raise ValueError(
                "Video frame-rate metadata is invalid; reliable timing cannot be established."
            )
        frame_count = max(0, capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sword = SwordSegmenter(sword_model) if sword_model is not None else None
        next_sample = 0.0
        index, previous_t = 0, -1.0
        with PoseExtractor(pose_model) as extractor:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                raw_t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                fallback = index / fps
                if not math.isfinite(raw_t) or (index > 0 and raw_t <= previous_t):
                    raw_t = fallback
                    if (
                        "Nominal-FPS timestamps were used because decoder presentation timestamps were unavailable."
                        not in warnings
                    ):
                        warnings.append(
                            "Nominal-FPS timestamps were used because decoder presentation timestamps were unavailable."
                        )
                if raw_t <= previous_t:
                    raise ValueError(
                        "Video timestamps are inconsistent; convert to a constant-frame-rate MP4."
                    )
                previous_t = raw_t
                index += 1
                if raw_t > config.max_seconds:
                    warnings.append(
                        f"Analysis stopped at the configured {config.max_seconds:g}-second duration limit."
                    )
                    break
                if raw_t + 1e-6 < next_sample:
                    continue
                next_sample = raw_t + 1 / config.sample_fps - 1e-5
                if config.mirrored_input:
                    bgr = cv2.flip(bgr, 1)
                pose = extractor.extract(bgr, raw_t)
                if sword is not None:
                    pose.swords = sword.detect(bgr)
                frames.append(pose)
                if progress is not None:
                    progress(
                        min(0.95, index / frame_count) if frame_count else 0,
                        f"Extracted {len(frames)} pose samples",
                    )
    finally:
        capture.release()
    if not frames:
        raise ValueError("No video frames could be decoded.")
    return frames, warnings


def annotate_video(source, frames, result, config, output_path):
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Could not reopen source video for annotation.")
    writer = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width, height = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("The local OpenCV build does not provide an MP4 video writer.")
        j, index = 0, 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(t) or (index and t <= 0):
                t = index / fps
            index += 1
            if t > frames[-1].t + 1 / fps:
                break
            while j + 1 < len(frames) and abs(frames[j + 1].t - t) < abs(frames[j].t - t):
                j += 1
            pose = (
                frames[j]
                if abs(frames[j].t - t) <= 0.07
                else PoseFrame(t, None, None, width, height, 0)
            )
            if config.mirrored_input:
                bgr = cv2.flip(bgr, 1)
            active = [a for a in result.actions if a.action.start <= t <= a.action.end]
            text = ACTION_NAMES[active[0].action.kind] if active else "No supported action label"
            if active:
                text += (
                    f" | {active[0].total:.1f}/100"
                    if active[0].total is not None
                    else " | score withheld"
                )
            writer.write(draw_overlay(bgr, pose, text))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return Path(output_path)


def analyze_video(
    path,
    output_dir,
    config=None,
    pose_model=DEFAULT_POSE_MODEL,
    sword_model=None,
    annotate=True,
    progress=None,
    source_name=None,
):
    config = config or AnalysisConfig()
    frames, capture_warnings = extract_video(path, config, pose_model, sword_model, progress)
    provenance = {
        "input_sha256": sha256(path),
        "pose_model_sha256": sha256(pose_model),
        "source_kind": "video",
        "timing": "Decoded video presentation timestamps, with explicitly reported nominal-FPS fallback.",
    }
    if sword_model is not None:
        provenance["sword_model_sha256"] = sha256(sword_model)
        provenance["sword_use"] = (
            "Observed segmentation regions only; sword-tip, contact, and blade-technique scoring are not implemented."
        )
    result, features = analyze_frames(
        frames, config, source_name or Path(path).name, provenance=provenance
    )
    result.warnings.extend(capture_warnings)
    artifacts = save_outputs(result, features, frames, output_dir)
    if annotate:
        try:
            artifacts["video"] = annotate_video(
                path, frames, result, config, Path(output_dir) / "annotated_video.mp4"
            )
        except (ValueError, RuntimeError) as exc:
            result.warnings.append(f"Annotated video unavailable: {exc}")
            artifacts.update(save_outputs(result, features, frames, output_dir))
    if progress is not None:
        progress(1.0, "Report ready")
    return result, artifacts


def analyze_camera(
    output_dir,
    config=None,
    camera_id=0,
    seconds=15,
    pose_model=DEFAULT_POSE_MODEL,
    sword_model=None,
    preview=None,
):
    """Time each webcam frame when it is captured.

    If pose tracking takes longer, there may be more time between samples.
    Keep those gaps so the analysis can tell when a movement was interrupted.
    """
    import cv2

    config = config or AnalysisConfig()
    if not 3 <= seconds <= min(120, config.max_seconds):
        raise ValueError(
            "Camera recording duration must be 3-120 seconds and within the session limit."
        )
    capture = cv2.VideoCapture(int(camera_id))
    if not capture.isOpened():
        raise ValueError(
            "Could not open the local webcam. Check its device ID and operating-system permission."
        )
    frames = []
    try:
        sword = SwordSegmenter(sword_model) if sword_model is not None else None
        previous_sample = -1.0
        with PoseExtractor(pose_model) as extractor:
            started = time.perf_counter()
            while time.perf_counter() - started < seconds:
                ok, bgr = capture.read()
                t = time.perf_counter() - started
                if not ok:
                    raise RuntimeError("The webcam stopped returning frames.")
                if t - previous_sample < 1 / config.sample_fps:
                    continue
                previous_sample = t
                if config.mirrored_input:
                    bgr = cv2.flip(bgr, 1)
                frame = extractor.extract(bgr, t)
                if sword is not None:
                    frame.swords = sword.detect(bgr)
                frames.append(frame)
                if preview is not None:
                    preview(
                        draw_overlay(
                            bgr, frame, "Hold en garde initially; PDF analysis follows capture"
                        ),
                        t,
                    )
    finally:
        capture.release()
    result, features = analyze_frames(
        frames,
        config,
        f"Webcam {camera_id}",
        provenance={
            "source_kind": "webcam",
            "timing": "Monotonic acquisition clock",
            "pose_model_sha256": sha256(pose_model),
        },
    )
    return result, save_outputs(result, features, frames, output_dir)


def analyze_landmark_file(path, output_dir, config=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0":
        raise ValueError("Unsupported landmark schema version.")
    frames = [PoseFrame.from_dict(item) for item in data["frames"]]
    result, features = analyze_frames(
        frames,
        config,
        data.get("source_name", Path(path).name),
        simulated=bool(data.get("simulated", False)),
        provenance={"source_kind": "landmark replay", "input_sha256": sha256(path)},
    )
    return result, save_outputs(result, features, frames, output_dir)
