"""Build the sample side-view and front-view reports."""

from pathlib import Path

from fencing_coach.analysis import analyze_frames
from fencing_coach.demo import synthetic_sequence
from fencing_coach.report import write_pdf


def main():
    root = Path(__file__).resolve().parents[1] / "examples"
    for name, label, yaw in (("side_view", "Side view", 0), ("frontal_view", "Front view", 90)):
        frames = synthetic_sequence(yaw_degrees=yaw, poor_trunk=True)
        result, _ = analyze_frames(
            frames,
            source_name=f"Sample lunge and recovery ({label.lower()})",
            simulated=True,
        )
        report = write_pdf(result, root / f"{name}.pdf", frames)
        print(report)


if __name__ == "__main__":
    main()
