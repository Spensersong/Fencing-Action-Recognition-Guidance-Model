"""Local MediaPipe Tasks extraction and optional user-owned YOLO segmentation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .schema import PoseFrame

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSE_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_SWORD_MODEL = ROOT / "best(1).pt"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PoseExtractor:
    def __init__(self, model_path=DEFAULT_POSE_MODEL, confidence=0.5):
        import mediapipe as mp

        self.mp = mp
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError("Pose model missing. Run: python tools/download_pose_model.py")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=2,
            min_pose_detection_confidence=confidence,
            min_pose_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.last_timestamp = -1

    def extract(self, bgr, t):
        import cv2

        timestamp = int(round(t * 1000))
        if timestamp <= self.last_timestamp:
            raise ValueError("MediaPipe requires strictly increasing timestamps in milliseconds.")
        self.last_timestamp = timestamp
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp)

        def array(landmarks):
            return np.array(
                [
                    [
                        lm.x,
                        lm.y,
                        lm.z,
                        lm.visibility if lm.visibility is not None else 0,
                        lm.presence if lm.presence is not None else 0,
                    ]
                    for lm in landmarks
                ],
                dtype=float,
            )

        people = len(result.pose_landmarks)
        return PoseFrame(
            float(t),
            array(result.pose_landmarks[0]) if people == 1 else None,
            array(result.pose_world_landmarks[0]) if people == 1 else None,
            bgr.shape[1],
            bgr.shape[0],
            people,
        )

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SwordSegmenter:
    """Expose the checkpoint's actual class/mask outputs. No invented anatomy."""

    def __init__(self, model_path=DEFAULT_SWORD_MODEL):
        from ultralytics import YOLO

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Sword weights not found: {model_path}")
        self.model = YOLO(str(model_path))
        if self.model.task != "segment":
            raise ValueError("Expected a YOLO segmentation checkpoint for this project.")
        self.names = self.model.names

    def detect(self, bgr):
        result = self.model.predict(bgr, conf=0.35, verbose=False, device="cpu")[0]
        detections = []
        if result.boxes is None:
            return detections
        for i, box in enumerate(result.boxes):
            cls = int(box.cls.item())
            polygon = (
                result.masks.xy[i].tolist()
                if result.masks is not None and i < len(result.masks.xy)
                else []
            )
            detections.append(
                {
                    "class_id": cls,
                    "label": self.names[cls],
                    "confidence": float(box.conf.item()),
                    "box_xyxy": box.xyxy[0].tolist(),
                    "polygon_xy": polygon,
                }
            )
        return detections


def draw_overlay(bgr, pose, text=""):
    import cv2

    from .report import EDGES

    output = bgr.copy()
    if pose.image is not None:
        points = pose.image[:, :2] * [pose.width, pose.height]
        for a, b in EDGES:
            if min(pose.image[a, 3], pose.image[b, 3]) >= 0.65:
                cv2.line(
                    output,
                    tuple(points[a].astype(int)),
                    tuple(points[b].astype(int)),
                    (120, 235, 100),
                    2,
                )
    for sword in pose.swords:
        polygon = np.asarray(sword.get("polygon_xy", []), dtype=np.int32)
        if len(polygon) >= 3:
            cv2.polylines(output, [polygon], True, (0, 200, 255), 2)
        x1, y1, x2, y2 = map(int, sword["box_xyxy"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.putText(
            output,
            f"{sword['label']} {sword['confidence']:.2f}",
            (max(0, x1), max(20, y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
        )
    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (25, 35, 45), -1)
    cv2.putText(
        output,
        f"{pose.t:.2f}s | {text}"[:110],
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return output
