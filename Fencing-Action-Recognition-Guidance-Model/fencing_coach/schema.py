"""Public data contracts shared by video, geometry, scoring, and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

ACTION_NAMES = {
    "en_garde": "En garde",
    "advance": "Advance",
    "retreat": "Retreat",
    "arm_extension": "Straight arm extension",
    "lunge": "Lunge",
    "recovery": "Recovery to en garde",
    "advance_lunge": "Advance-lunge",
    "unknown": "Unclassified movement",
}

DIMENSIONS = {
    "technique": ("Joint configuration", 45),
    "timing": ("Timing and coordination", 25),
    "control": ("Movement control", 20),
    "presentation": ("Fluency and consistency", 10),
}


@dataclass(frozen=True)
class AnalysisConfig:
    weapon_side: str = "right"
    front_side: str = "right"
    weapon: str = "foil"
    visibility_threshold: float = 0.65
    calibration_seconds: float = 1.2
    max_gap_seconds: float = 0.20
    min_action_seconds: float = 0.16
    min_coverage: float = 0.80
    camera_static: bool = True
    camera_level: bool = True
    mirrored_input: bool = False
    sample_fps: float = 30.0
    max_seconds: float = 300.0

    def __post_init__(self):
        if self.weapon_side not in {"left", "right"} or self.front_side not in {"left", "right"}:
            raise ValueError("Weapon side and front side must be left or right.")
        if self.weapon not in {"foil", "epee", "sabre"}:
            raise ValueError("Weapon must be foil, epee, or sabre.")
        if not 0.3 <= self.visibility_threshold <= 1:
            raise ValueError("Visibility threshold must be between 0.3 and 1.")
        if not 5 <= self.sample_fps <= 60 or not 2 <= self.max_seconds <= 1800:
            raise ValueError("Sample FPS must be 5-60; duration must be 2-1800 seconds.")
        if not 0.5 <= self.min_coverage <= 1 or self.calibration_seconds < 0.5:
            raise ValueError("Invalid coverage or calibration duration.")
        if self.max_gap_seconds <= 0 or self.min_action_seconds <= 0:
            raise ValueError("Temporal limits must be positive.")


@dataclass
class PoseFrame:
    """Landmarks are (33, 5): x, y, z, visibility, presence.

    Image x/y are normalized by width/height. World x/y/z are estimated metres
    relative to the hips. Hip-relative world positions cannot measure locomotion.
    """

    t: float
    image: np.ndarray | None
    world: np.ndarray | None
    width: int = 1280
    height: int = 720
    people: int = 1
    swords: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        def clean_array(value):
            return (
                value.tolist()
                if value is not None and value.shape == (33, 5) and np.isfinite(value).all()
                else None
            )

        return {
            "t": self.t,
            "image": clean_array(self.image),
            "world": clean_array(self.world),
            "width": self.width,
            "height": self.height,
            "people": self.people,
            "swords": self.swords,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            t=float(data["t"]),
            image=np.asarray(data["image"], dtype=float) if data.get("image") is not None else None,
            world=np.asarray(data["world"], dtype=float) if data.get("world") is not None else None,
            width=int(data.get("width", 1280)),
            height=int(data.get("height", 720)),
            people=int(data.get("people", 1)),
            swords=data.get("swords", []),
        )


@dataclass
class Calibration:
    valid: bool
    reason: str
    end: float = 0.0
    axis: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    screen_axis: list[float] = field(default_factory=lambda: [1.0, 0.0])
    leg_length: float = 1.0
    screen_scale: float = 1.0
    baseline: dict[str, float] = field(default_factory=dict)
    bone_lengths: list[float] = field(default_factory=list)
    view: str = "unknown"
    projection: float = 0.0
    movement_observable: bool = False


@dataclass
class FeatureFrame:
    t: float
    valid: bool
    values: dict[str, float | None] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    view: str = "unknown"
    quality: float = 0.0


@dataclass
class Action:
    kind: str
    start: float
    end: float
    peak: float
    complete: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.start, self.end, self.peak = float(self.start), float(self.end), float(self.peak)
        self.complete = bool(self.complete)


@dataclass
class MetricResult:
    key: str
    label: str
    dimension: str
    weight: float
    value: float | None
    unit: str
    target: list[float]
    score: float | None
    observed_at: float | None
    reason: str
    cue: str
    source: str


@dataclass
class ActionResult:
    action: Action
    metrics: list[MetricResult]
    dimensions: dict[str, float | None]
    coverage: float
    total: float | None
    status: str
    notes: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    schema_version: str
    software_version: str
    rubric_version: str
    source_name: str
    simulated: bool
    config: dict[str, Any]
    calibration: Calibration
    duration: float
    observed_fraction: float
    actions: list[ActionResult]
    warnings: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
