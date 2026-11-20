import copy
import json

import numpy as np
import pytest

from fencing_coach.analysis import analyze_frames
from fencing_coach.demo import synthetic_sequence
from fencing_coach.geometry import angle
from fencing_coach.pipeline import save_outputs
from fencing_coach.schema import AnalysisConfig, PoseFrame
from fencing_coach.scoring import load_rubric, range_score


def select(result, kind):
    return next(a for a in result.actions if a.action.kind == kind)


@pytest.mark.parametrize("kind", ["advance", "retreat", "arm_extension", "lunge", "advance_lunge"])
def test_named_temporal_drills_are_recognized(kind):
    result, _ = analyze_frames(synthetic_sequence(kind))
    action = select(result, kind)
    assert action.action.complete
    assert action.total is not None
    assert 0 <= action.total <= 100


@pytest.mark.parametrize("yaw", [0, 15, 30, 45, 65, 90, 145, 180])
def test_known_3d_knee_geometry_is_invariant_under_camera_yaw(yaw):
    result, features = analyze_frames(synthetic_sequence(yaw_degrees=yaw))
    assert result.calibration.valid
    sample = min(features, key=lambda f: abs(f.t - 2.75))
    # Independently specified links: L=0.45, endpoint=(0.32, 0.55).
    expected = np.degrees(np.arccos((2 * 0.45**2 - 0.32**2 - 0.55**2) / (2 * 0.45**2)))
    assert sample.values["front_knee"] == pytest.approx(expected, abs=0.01)
    assert select(result, "lunge").action.complete


def test_left_handed_and_right_handed_drills_are_equivalent():
    right, _ = analyze_frames(synthetic_sequence())
    left, _ = analyze_frames(
        synthetic_sequence(side="left"), AnalysisConfig(front_side="left", weapon_side="left")
    )
    a, b = select(right, "lunge"), select(left, "lunge")
    assert a.total == b.total
    assert a.action.peak == b.action.peak


@pytest.mark.parametrize("width,height", [(720, 1280), (1920, 1080), (1000, 1000)])
def test_aspect_ratio_does_not_change_3d_joint_measurements(width, height):
    result, _ = analyze_frames(synthetic_sequence(width=width, height=height))
    reference, _ = analyze_frames(synthetic_sequence())
    assert select(result, "lunge").metrics[0].value == pytest.approx(
        select(reference, "lunge").metrics[0].value
    )


def test_frontal_view_withholds_translation_and_full_score():
    result, _ = analyze_frames(synthetic_sequence(yaw_degrees=90))
    assert not result.calibration.movement_observable
    action = select(result, "lunge")
    assert action.total is None
    assert next(m for m in action.metrics if m.key == "hip_range").value is None
    step, _ = analyze_frames(synthetic_sequence("advance", yaw_degrees=90))
    assert all(a.action.kind != "advance" for a in step.actions)


def test_missing_arm_is_not_filled_with_an_ideal_elbow():
    frames = synthetic_sequence()
    for frame in frames:
        if frame.t > 1.4:
            frame.image[14, 3:5] = 0.1
    result, _ = analyze_frames(frames)
    lunge = select(result, "lunge")
    metric = next(m for m in lunge.metrics if m.key == "elbow")
    assert metric.value is None and metric.score is None
    assert lunge.total is None
    assert lunge.dimensions["technique"] is None


def test_occlusion_and_missing_frames_break_action_intervals():
    frames = synthetic_sequence()
    for frame in frames:
        if 2.25 < frame.t < 2.75:
            frame.image = frame.world = None
    result, _ = analyze_frames(frames)
    assert all(not (a.action.start < 2.25 and a.action.end > 2.75) for a in result.actions)
    assert result.observed_fraction < 1
    assert all(a.total is None for a in result.actions if a.action.kind == "lunge")


def test_entirely_missing_or_multiple_people_never_gets_a_score():
    for frames in (
        [PoseFrame(t / 30, None, None, people=0) for t in range(100)],
        synthetic_sequence(),
    ):
        if frames[0].image is not None:
            for frame in frames:
                frame.people = 2
        result, _ = analyze_frames(frames)
        assert not result.calibration.valid
        assert not result.actions


def test_camera_motion_disables_locomotion_claims():
    result, _ = analyze_frames(synthetic_sequence("advance"), AnalysisConfig(camera_static=False))
    assert not result.calibration.movement_observable
    assert all(a.action.kind not in {"advance", "retreat"} for a in result.actions)


def test_incomplete_cutoff_does_not_receive_overall_score():
    frames = [f for f in synthetic_sequence() if f.t <= 2.45]
    result, _ = analyze_frames(frames)
    assert all(a.total is None for a in result.actions if a.action.kind == "lunge")


def test_trunk_deviation_is_measured_and_linked_to_fixed_cue():
    result, _ = analyze_frames(synthetic_sequence(poor_trunk=True))
    action = select(result, "lunge")
    trunk = next(m for m in action.metrics if m.key == "trunk")
    assert trunk.value > 22
    assert trunk.score < 100
    assert trunk.observed_at == action.action.peak
    assert trunk.source == "FIE_LUNGE"
    assert "short lunge" in trunk.cue
    assert action.total < 100


def test_invalid_timestamp_order_is_rejected():
    frames = synthetic_sequence()
    frames[10].t = frames[9].t
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_frames(frames)


def test_degenerate_geometry_is_not_an_angle():
    assert angle(np.zeros(3), np.zeros(3), np.ones(3)) is None
    assert angle(np.array([1, 0, 0]), np.zeros(3), np.array([0, 1, 0])) == pytest.approx(90)


def test_range_scoring_is_bounded_and_uses_independent_reference_examples():
    assert range_score(90, [80, 120], [40, 160]) == 100
    assert range_score(60, [80, 120], [40, 160]) == 50
    assert range_score(140, [80, 120], [40, 160]) == 50
    assert range_score(200, [80, 120], [40, 160]) == 0


def test_every_profile_has_exact_dimension_weights():
    expected = {"technique": 45, "timing": 25, "control": 20, "presentation": 10}
    for profile in load_rubric()["profiles"].values():
        actual = {key: sum(m[3] for m in profile if m[2] == key) for key in expected}
        assert actual == expected


def test_json_pdf_csv_export_and_replay_are_consistent(tmp_path):
    from pypdf import PdfReader

    from fencing_coach.pipeline import analyze_landmark_file

    frames = synthetic_sequence(poor_trunk=True)
    result, features = analyze_frames(
        frames, source_name="Synthetic evidence fixture", simulated=True
    )
    artifacts = save_outputs(result, features, frames, tmp_path / "first")
    payload = json.loads(artifacts["json"].read_text())
    assert payload["simulated"] is True
    assert isinstance(payload["actions"][0]["action"]["complete"], bool)
    reader = PdfReader(artifacts["pdf"])
    text = " ".join(page.extract_text() for page in reader.pages)
    assert "SIMULATED DATA" in text
    assert "Trunk inclination at peak" in text
    assert "FIE_LUNGE" in text
    assert "time_seconds" in artifacts["csv"].read_text()
    replay, _ = analyze_landmark_file(artifacts["landmarks"], tmp_path / "replay")
    assert replay.simulated
    assert select(replay, "lunge").total == select(result, "lunge").total


def test_bad_landmark_data_is_excluded_not_scored():
    frames = copy.deepcopy(synthetic_sequence())
    for f in frames:
        f.world[25, 0] = np.nan
    result, _ = analyze_frames(frames)
    assert not result.calibration.valid
    assert not result.actions


def test_invalid_arrays_can_be_exported_as_unavailable_observations(tmp_path):
    frames = synthetic_sequence()
    for f in frames:
        f.world[25, 0] = np.nan
    result, features = analyze_frames(frames, simulated=True)
    outputs = save_outputs(result, features, frames, tmp_path)
    data = json.loads(outputs["landmarks"].read_text())
    assert all(f["world"] is None for f in data["frames"])
    assert "NaN" not in outputs["landmarks"].read_text()


def test_sample_rate_changes_timing_resolution_without_inventing_onset_order():
    fast, _ = analyze_frames(synthetic_sequence(fps=30))
    slow, _ = analyze_frames(synthetic_sequence(fps=15))
    a, b = select(fast, "lunge"), select(slow, "lunge")
    assert a.action.peak == pytest.approx(b.action.peak, abs=0.12)
    assert b.total is None or 0 <= b.total <= 100


def test_recovery_does_not_label_the_entire_peak_hold_as_recovering():
    result, _ = analyze_frames(synthetic_sequence())
    assert select(result, "recovery").action.start >= 2.90


def test_sabre_does_not_receive_unvalidated_classical_arm_grades():
    result, _ = analyze_frames(synthetic_sequence(), AnalysisConfig(weapon="sabre"))
    action = select(result, "lunge")
    assert action.total is None
    assert all(
        m.score is None
        for m in action.metrics
        if m.key in {"elbow", "arm_lead", "reach_efficiency"}
    )
