"""Evaluate actual analysis.json outputs against an exhaustively labelled CSV."""

import argparse
import csv
import json
from pathlib import Path

from fencing_coach.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    with args.labels.open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    for row in labels:
        row["start"], row["end"] = float(row["start"]), float(row["end"])
        row["coach_score"] = float(row["coach_score"]) if row.get("coach_score") else None
    predictions = {}
    for path in args.predictions.rglob("analysis.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        video_id = result["source_name"]
        if video_id in predictions:
            raise ValueError(f"Duplicate video_id in predictions: {video_id}")
        predictions[video_id] = result
    result = evaluate(labels, predictions, args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
