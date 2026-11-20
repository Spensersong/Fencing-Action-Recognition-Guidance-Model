"""Interval-level held-out evaluation. This does not tune or train the recognizer."""

from collections import defaultdict
from math import isfinite

from .schema import ACTION_NAMES


def interval_iou(a, b):
    intersection = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return intersection / union if union > 0 else 0


def evaluate(labels, predictions, iou_threshold=0.5):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be greater than 0 and at most 1.")
    truth_by_video = defaultdict(list)
    for row in labels:
        if row["action"] not in ACTION_NAMES or row["action"] == "unknown":
            raise ValueError("Labels must use the supported action taxonomy.")
        if (
            not isfinite(row["start"])
            or not isfinite(row["end"])
            or row["end"] <= row["start"]
            or not row.get("view")
        ):
            raise ValueError("Each label needs an observed view and an increasing time interval.")
        if row.get("coach_score") is not None and not 0 <= row["coach_score"] <= 100:
            raise ValueError("Coach scores must be between 0 and 100.")
        truth_by_video[row["video_id"]].append(row)
    if set(predictions) - set(truth_by_video):
        raise ValueError(
            "Every prediction video must be exhaustively annotated in the labels file."
        )
    groups = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "score_errors": [], "matched": 0, "scored": 0}
    )
    for video_id, truth in truth_by_video.items():
        views = {row["view"] for row in truth}
        if len(views) != 1:
            raise ValueError("Use a separate video_id for every camera-view segment.")
        view = next(iter(views))
        result = predictions.get(video_id)
        if result and result.get("simulated"):
            raise ValueError(
                "Synthetic results cannot be included in real-world held-out evaluation."
            )
        observed = (
            [a for a in result.get("actions", []) if a["action"]["kind"] != "unknown"]
            if result
            else []
        )
        candidates = []
        for i, label in enumerate(truth):
            for j, pred in enumerate(observed):
                if label["action"] == pred["action"]["kind"]:
                    overlap = interval_iou(label, pred["action"])
                    if overlap >= iou_threshold:
                        candidates.append((overlap, i, j))
        seen_truth, seen_pred = set(), set()
        for _, i, j in sorted(candidates, reverse=True):
            if i in seen_truth or j in seen_pred:
                continue
            seen_truth.add(i)
            seen_pred.add(j)
            label, pred = truth[i], observed[j]
            for key in ("all", f"view:{view}", f"action:{label['action']}"):
                group = groups[key]
                group["tp"] += 1
                group["matched"] += 1
                group["scored"] += pred.get("total") is not None
                if pred.get("total") is not None and label.get("coach_score") is not None:
                    group["score_errors"].append(abs(pred["total"] - label["coach_score"]))
        for i, label in enumerate(truth):
            if i not in seen_truth:
                for key in ("all", f"view:{view}", f"action:{label['action']}"):
                    groups[key]["fn"] += 1
        for j, pred in enumerate(observed):
            if j not in seen_pred:
                for key in ("all", f"view:{view}", f"action:{pred['action']['kind']}"):
                    groups[key]["fp"] += 1
    summary = {}
    for key, values in groups.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        summary[key] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "matched_action_score_coverage": values["scored"] / values["matched"]
            if values["matched"]
            else None,
            "coach_score_pairs": len(values["score_errors"]),
            "coach_score_mae": sum(values["score_errors"]) / len(values["score_errors"])
            if values["score_errors"]
            else None,
        }
    return {
        "iou_threshold": iou_threshold,
        "videos": len(truth_by_video),
        "groups": summary,
        "interpretation": "Recognition and scoring coverage are reported separately. Missing or rejected predictions count as false negatives. Small samples do not establish generalization.",
    }
