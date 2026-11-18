import pytest

from fencing_coach.evaluation import evaluate, interval_iou


def label(action="lunge", view="side"):
    return {
        "video_id": "athlete01.mp4",
        "view": view,
        "action": action,
        "start": 2.0,
        "end": 3.0,
        "coach_score": 80.0,
    }


def prediction(kind="lunge", score=75.0):
    return {
        "source_name": "athlete01.mp4",
        "simulated": False,
        "actions": [{"action": {"kind": kind, "start": 2.1, "end": 3.1}, "total": score}],
    }


def test_interval_overlap_has_independent_expected_value():
    assert interval_iou({"start": 0, "end": 2}, {"start": 1, "end": 3}) == pytest.approx(1 / 3)


def test_matching_and_coach_error_are_reported():
    result = evaluate([label()], {"athlete01.mp4": prediction()})
    group = result["groups"]["all"]
    assert group["precision"] == group["recall"] == 1
    assert group["coach_score_mae"] == 5
    assert group["coach_score_pairs"] == 1


def test_wrong_label_is_false_positive_and_false_negative():
    result = evaluate([label()], {"athlete01.mp4": prediction("advance")})
    group = result["groups"]["all"]
    assert group["false_positives"] == group["false_negatives"] == 1
    assert group["true_positives"] == 0


def test_rejected_video_stays_in_recall_denominator():
    result = evaluate([label()], {})
    assert result["groups"]["all"]["false_negatives"] == 1
    assert result["groups"]["all"]["recall"] == 0
    assert result["groups"]["all"]["precision"] is None


def test_withheld_score_is_not_treated_as_zero_or_dropped_silently():
    result = evaluate([label()], {"athlete01.mp4": prediction(score=None)})
    assert result["groups"]["all"]["matched_action_score_coverage"] == 0
    assert result["groups"]["all"]["coach_score_mae"] is None


def test_duplicate_predictions_do_not_match_one_label_twice():
    data = prediction()
    data["actions"] *= 2
    result = evaluate([label()], {"athlete01.mp4": data})
    assert result["groups"]["all"]["true_positives"] == 1
    assert result["groups"]["all"]["false_positives"] == 1


def test_synthetic_results_cannot_be_used_as_a_real_world_benchmark():
    data = prediction()
    data["simulated"] = True
    with pytest.raises(ValueError, match="Synthetic"):
        evaluate([label()], {"athlete01.mp4": data})
