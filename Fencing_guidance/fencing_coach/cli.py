"""Command-line entry point. Importing the package never opens a camera."""

import argparse
from pathlib import Path

from .analysis import analyze_frames
from .demo import synthetic_sequence
from .pipeline import analyze_camera, analyze_landmark_file, analyze_video, save_outputs
from .schema import ACTION_NAMES, AnalysisConfig
from .vision import DEFAULT_POSE_MODEL


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyse solo fencing drills and export a coaching PDF."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int, metavar="DEVICE_ID")
    source.add_argument("--landmarks", type=Path)
    source.add_argument(
        "--demo",
        action="store_true",
        help="Run a sample drill with simulated landmarks",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/session"))
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--sword-model", type=Path, help="Optional trusted local YOLO segmentation weights"
    )
    parser.add_argument("--weapon-hand", choices=["left", "right"], default="right")
    parser.add_argument("--front-leg", choices=["left", "right"], default="right")
    parser.add_argument("--weapon", choices=["foil", "epee", "sabre"], default="foil")
    parser.add_argument("--seconds", type=float, default=15, help="Webcam capture duration")
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--sample-fps", type=float, default=30)
    parser.add_argument(
        "--moving-camera", action="store_true", help="Withhold translation-dependent conclusions"
    )
    parser.add_argument(
        "--camera-not-level",
        action="store_true",
        help="Capture cannot support the configured reference frame",
    )
    parser.add_argument(
        "--mirrored-input", action="store_true", help="Unmirror the source before inference"
    )
    parser.add_argument("--no-annotated-video", action="store_true")
    args = parser.parse_args(argv)
    config = AnalysisConfig(
        weapon_side=args.weapon_hand,
        front_side=args.front_leg,
        weapon=args.weapon,
        camera_static=not args.moving_camera,
        camera_level=not args.camera_not_level,
        mirrored_input=args.mirrored_input,
        sample_fps=args.sample_fps,
        max_seconds=args.max_seconds,
    )
    try:
        if args.demo:
            result, features = analyze_frames(
                synthetic_sequence(side=args.front_leg),
                config,
                "Sample lunge and recovery",
                simulated=True,
            )
            artifacts = save_outputs(
                result, features, synthetic_sequence(side=args.front_leg), args.output
            )
        elif args.video:
            result, artifacts = analyze_video(
                args.video,
                args.output,
                config,
                args.pose_model,
                args.sword_model,
                not args.no_annotated_video,
            )
        elif args.camera is not None:
            result, artifacts = analyze_camera(
                args.output, config, args.camera, args.seconds, args.pose_model, args.sword_model
            )
        else:
            result, artifacts = analyze_landmark_file(args.landmarks, args.output, config)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Analysis could not be completed: {exc}\n")
    print(
        f"Calibration: {'accepted' if result.calibration.valid else 'rejected'} | View: {result.calibration.view}"
    )
    for item in result.actions:
        score = (
            f"{item.total:.1f}/100"
            if item.total is not None
            else f"withheld ({item.coverage:.0%} weighted evidence)"
        )
        print(
            f"{item.action.start:.2f}-{item.action.end:.2f}s {ACTION_NAMES[item.action.kind]}: {score}"
        )
    print(f"PDF: {artifacts['pdf'].resolve()}")
    return 0


if __name__ == "__main__":
    main()
