"""Rebuild explicitly synthetic, English coaching-report examples."""

import shutil
from pathlib import Path

from fencing_coach.analysis import analyze_frames
from fencing_coach.demo import synthetic_sequence
from fencing_coach.pipeline import save_outputs


def main():
    root = Path(__file__).resolve().parents[1] / "examples"
    for name, yaw in (("side_view", 0), ("frontal_view", 90)):
        frames = synthetic_sequence(yaw_degrees=yaw, poor_trunk=True)
        result, features = analyze_frames(
            frames,
            source_name=f"Analytical synthetic lunge with a specified trunk deviation ({name})",
            simulated=True,
            provenance={
                "source_kind": "analytical synthetic fixture",
                "purpose": "Software demonstration; not real-world validation or an athlete assessment.",
            },
        )
        artifacts = save_outputs(result, features, frames, root / name)
        if name == "side_view":
            shutil.copyfile(artifacts["pdf"], root / "Fencing_Coaching_Report_Example.pdf")
        print(f"Built {name}: {artifacts['pdf']}")


if __name__ == "__main__":
    main()
