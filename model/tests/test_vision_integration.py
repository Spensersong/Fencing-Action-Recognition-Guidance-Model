"""Integration tests for the pose model, sword model, and video codec."""

import os
from pathlib import Path

import pytest

from fencing_coach.schema import AnalysisConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("FENCING_VISION_TESTS") != "1",
    reason="Set FENCING_VISION_TESTS=1 to run actual model and codec integration.",
)


def test_real_pose_model_rejects_blank_video_and_exports_pdf(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pytest.importorskip("mediapipe")
    from pypdf import PdfReader

    from fencing_coach.pipeline import analyze_video

    video = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 240))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    result, artifacts = analyze_video(video, tmp_path / "report", AnalysisConfig(sample_fps=15))
    assert not result.calibration.valid
    assert not result.actions
    assert not result.simulated
    assert len(PdfReader(artifacts["pdf"]).pages) >= 2
    assert artifacts["video"].stat().st_size > 0
    decoded = cv2.VideoCapture(str(artifacts["video"]))
    ok, image = decoded.read()
    decoded.release()
    assert ok and image.shape == (240, 320, 3)


def test_original_sword_checkpoint_loads_and_predicts():
    pytest.importorskip("ultralytics")
    import numpy as np

    from fencing_coach.vision import DEFAULT_SWORD_MODEL, SwordSegmenter, sha256

    assert Path(DEFAULT_SWORD_MODEL).is_file()
    assert (
        sha256(DEFAULT_SWORD_MODEL)
        == "56822aa204bf1b2a36b4904a11c1c8d1dfa1483d4e4a7800cc4f9be55a1b34e4"
    )
    segmenter = SwordSegmenter()
    assert segmenter.names == {0: "sword"}
    assert segmenter.model.task == "segment"
    output = segmenter.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert isinstance(output, list)
    assert all(item["label"] == "sword" for item in output)
