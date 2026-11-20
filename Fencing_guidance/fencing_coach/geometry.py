"""Measure joint angles and distances from the body landmarks."""

import numpy as np

SIDES = {
    "left": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "hip": 23,
        "knee": 25,
        "ankle": 27,
        "heel": 29,
        "toe": 31,
    },
    "right": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "hip": 24,
        "knee": 26,
        "ankle": 28,
        "heel": 30,
        "toe": 32,
    },
}
CORE = [11, 12, 23, 24, 25, 26, 27, 28]
BONES = [(23, 25), (25, 27), (24, 26), (26, 28), (11, 23), (12, 24)]


def other(side):
    return "left" if side == "right" else "right"


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-7:
        return None
    return vector / length


def angle(a, b, c):
    ba, bc = unit(np.asarray(a) - b), unit(np.asarray(c) - b)
    if ba is None or bc is None:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc), -1.0, 1.0))))


def visible(frame, indices, threshold):
    if frame.image is None or frame.world is None or frame.people != 1:
        return False
    if frame.width <= 0 or frame.height <= 0:
        return False
    for arr in (frame.image, frame.world):
        if arr.shape != (33, 5) or not np.isfinite(arr[indices]).all():
            return False
        if np.min(arr[indices, 3:5]) < threshold:
            return False
    xy = frame.image[indices, :2]
    return bool(np.all(xy >= -0.015) and np.all(xy <= 1.015))


def pixels(frame):
    # x is a fraction of the width and y is a fraction of the height. Convert both to pixels.
    return frame.image[:, :2] * [frame.width, frame.height]


def bone_lengths(world):
    return np.array([np.linalg.norm(world[a, :3] - world[b, :3]) for a, b in BONES])


def view_for_axis(axis):
    projection = float(abs(axis[0]))
    if projection >= 0.82:
        return "side", projection
    if projection >= 0.45:
        return "oblique", projection
    return "frontal", projection


def measurements(frame, config, axis, leg_length, screen_axis, screen_scale):
    w, p = frame.world[:, :3], pixels(frame)
    front, rear, arm = (
        SIDES[config.front_side],
        SIDES[other(config.front_side)],
        SIDES[config.weapon_side],
    )
    hip = (w[23] + w[24]) / 2
    shoulder = (w[11] + w[12]) / 2
    spine = unit(shoulder - hip)
    values = {
        "front_knee": angle(w[front["hip"]], w[front["knee"]], w[front["ankle"]]),
        "rear_knee": angle(w[rear["hip"]], w[rear["knee"]], w[rear["ankle"]]),
        "trunk": float(np.degrees(np.arccos(np.clip(-spine[1], -1, 1))))
        if spine is not None
        else None,
        "stance": float(np.dot(w[front["ankle"]] - w[rear["ankle"]], axis) / leg_length),
        "hip_screen": float(np.dot((p[23] + p[24]) / 2, screen_axis) / screen_scale),
        "front_screen": float(np.dot(p[front["ankle"]], screen_axis) / screen_scale),
        "rear_screen": float(np.dot(p[rear["ankle"]], screen_axis) / screen_scale),
        "hip_height": float(((p[23] + p[24]) / 2)[1] / screen_scale),
        "knee_offset": float(abs(np.dot(w[front["knee"]] - w[front["ankle"]], axis)) / leg_length),
        "elbow": None,
        "reach": None,
    }
    if visible(frame, [arm["shoulder"], arm["elbow"], arm["wrist"]], config.visibility_threshold):
        arm_length = np.linalg.norm(w[arm["elbow"]] - w[arm["shoulder"]]) + np.linalg.norm(
            w[arm["wrist"]] - w[arm["elbow"]]
        )
        values["elbow"] = angle(w[arm["shoulder"]], w[arm["elbow"]], w[arm["wrist"]])
        if arm_length > 0.1:
            values["reach"] = float(np.dot(w[arm["wrist"]] - w[arm["shoulder"]], axis) / arm_length)
    return values
