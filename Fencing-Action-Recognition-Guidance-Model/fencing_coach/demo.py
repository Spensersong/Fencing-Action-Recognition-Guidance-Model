"""Analytical synthetic fixtures. Never presented as labelled training videos."""

from __future__ import annotations

import numpy as np

from .geometry import SIDES
from .schema import PoseFrame


def ramp(t, start, end):
    a = float(np.clip((t - start) / (end - start), 0, 1))
    return a * a * (3 - 2 * a)


def leg_knee(hip, ankle, length=0.45):
    """Two-link inverse kinematics with fixed, independently specified lengths."""
    delta = ankle - hip
    distance = float(np.linalg.norm(delta))
    if not 0 < distance <= 2 * length:
        raise ValueError("Synthetic leg target is unreachable.")
    perpendicular = np.array([delta[1], -delta[0], 0.0]) / np.linalg.norm(delta[:2])
    return (hip + ankle) / 2 + perpendicular * np.sqrt(max(0, length**2 - (distance / 2) ** 2))


def synthetic_sequence(
    kind="lunge", yaw_degrees=0, fps=30, side="right", width=1280, height=720, poor_trunk=False
):
    """Generate specified kinematics to test invariance and rejection behaviour.

    This tests the rule engine with ideal coordinates, NOT MediaPipe accuracy.
    Rotating the true coordinates does not simulate camera-dependent pose errors.
    """
    duration = 6.0 if kind == "advance_lunge" else 4.8
    output = []
    theta = np.radians(yaw_degrees)
    rotation = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    rear_side = "left" if side == "right" else "right"
    for t in np.arange(0, duration, 1 / fps):
        front_step = rear_step = lunge = extension = 0.0
        if kind in {"advance", "retreat", "advance_lunge"}:
            if kind == "retreat":
                rear_step = -0.22 * ramp(t, 1.8, 2.15)
                front_step = -0.22 * ramp(t, 2.06, 2.41)
            else:
                front_step = 0.22 * ramp(t, 1.8, 2.15)
                rear_step = 0.22 * ramp(t, 2.06, 2.41)
        if kind in {"lunge", "advance_lunge"}:
            offset = 1.0 if kind == "advance_lunge" else 0
            lunge = ramp(t, 2.10 + offset, 2.60 + offset) * (
                1 - ramp(t, 2.95 + offset, 3.55 + offset)
            )
            extension = ramp(t, 1.78 + offset, 2.16 + offset) * (
                1 - ramp(t, 3.0 + offset, 3.55 + offset)
            )
        elif kind == "arm_extension":
            extension = ramp(t, 1.8, 2.25) * (1 - ramp(t, 2.7, 3.15))
        root_x = (front_step + rear_step) / 2 + 0.35 * lunge
        root_y = 0.17 * lunge
        xyz = np.zeros((33, 3), dtype=float)
        for name, z in ((side, -0.10), (rear_side, 0.10)):
            ids = SIDES[name]
            hip = np.array([0.0, 0.0, z])
            if name == side:
                ankle = np.array(
                    [
                        0.35 + front_step - (front_step + rear_step) / 2 - 0.03 * lunge,
                        0.72 - 0.17 * lunge,
                        z,
                    ]
                )
            else:
                ankle = np.array(
                    [
                        -0.35 + rear_step - (front_step + rear_step) / 2 - 0.35 * lunge,
                        0.72 - 0.17 * lunge,
                        z,
                    ]
                )
            shoulder = hip + np.array([0.22 * lunge if poor_trunk else 0.0, -0.5, 0.0])
            xyz[ids["hip"]], xyz[ids["ankle"]], xyz[ids["shoulder"]] = hip, ankle, shoulder
            xyz[ids["knee"]] = leg_knee(hip, ankle)
            xyz[ids["heel"]] = ankle + [-0.05, 0.03, 0]
            xyz[ids["toe"]] = ankle + [0.15 if name == side else 0.03, 0.03, 0]
            if name == side:
                phi1 = np.radians(60 * (1 - extension))
                phi2 = np.radians(-10 * (1 - extension))
                elbow = shoulder + 0.3 * np.array([np.cos(phi1), np.sin(phi1), 0])
                wrist = elbow + 0.3 * np.array([np.cos(phi2), np.sin(phi2), 0])
            else:
                elbow = shoulder + [-0.15, 0.22, 0]
                wrist = elbow + [-0.12, -0.18, 0]
            xyz[ids["elbow"]], xyz[ids["wrist"]] = elbow, wrist
        xyz[0:11] = [0, -0.72, 0]
        xyz[17:23] = xyz[SIDES[side]["wrist"]]
        xyz = xyz @ rotation.T
        root = np.array([root_x, root_y, 0]) @ rotation.T
        world = np.column_stack([xyz, np.full((33, 2), 0.99)])
        image = world.copy()
        scale = min(width / 5.0, height / 2.5)
        image[:, 0] = (width / 2 + (xyz[:, 0] + root[0]) * scale) / width
        image[:, 1] = (height * 0.43 + (xyz[:, 1] + root[1]) * scale) / height
        image[:, 2] = (xyz[:, 2] + root[2]) * scale / width
        output.append(PoseFrame(float(t), image, world, width, height))
    return output
