"""Find basic solo drills by checking how the joints move through the clip.

The labels are for practice. Contact and bout scoring need other evidence.
"""

from __future__ import annotations

import numpy as np

from .features import continuous_runs, smooth_run
from .schema import Action


def spans(mask, times, merge_gap=0.0):
    intervals = []
    start = None
    last = None
    for i, yes in enumerate(mask):
        if yes:
            if start is None:
                start = i
            last = i
        elif start is not None and times[i] - times[last] > merge_gap:
            intervals.append((start, last))
            start = last = None
    if start is not None:
        intervals.append((start, last))
    return intervals


def velocity(values, times):
    if len(times) < 2:
        return np.zeros(len(times))
    return np.gradient(np.asarray(values, dtype=float), times)


def recognize(features, calibration, config):
    if not calibration.valid:
        return []
    actions = []
    for original in continuous_runs(features, config.max_gap_seconds):
        run = smooth_run(original)
        if len(run) < 5:
            continue
        t = np.array([f.t for f in run])

        def v(key, sequence=run):
            return np.array(
                [f.values.get(key) if f.values.get(key) is not None else np.nan for f in sequence]
            )

        stance, front_knee, rear_knee = v("stance"), v("front_knee"), v("rear_knee")
        base = calibration.baseline
        eligible = t >= calibration.end
        used = ~eligible.copy()
        run_actions = []

        # Recognise the lunge before comparing its form with the scoring ranges.
        lunge_mask = (
            eligible
            & (stance > base["stance"] + 0.18)
            & (front_knee < min(156, base["front_knee"] - 5))
            & ((rear_knee - front_knee > 12) | (rear_knee > base["rear_knee"] + 8))
        )
        for a, b in spans(lunge_mask, t, merge_gap=0.20):
            if t[b] - t[a] < config.min_action_seconds:
                continue
            start = a
            while (
                start > 0
                and t[a] - t[start - 1] < 0.75
                and t[start - 1] >= calibration.end
                and stance[start - 1] > base["stance"] + 0.035
            ):
                start -= 1
            start = max(int(np.searchsorted(t, calibration.end)), start - 2)
            # Look back for the arm to start while the feet are still close to guard.
            lookback = max(start - 1, 0)
            while (
                lookback > 0
                and t[start] - t[lookback - 1] < 0.40
                and t[lookback - 1] >= calibration.end
            ):
                if abs(stance[lookback - 1] - base["stance"]) > 0.08:
                    break
                lookback -= 1
            start = lookback
            peak = a + int(np.argmax(stance[a : b + 1]))
            plateau = np.flatnonzero(stance[a : b + 1] >= stance[peak] - 0.035)
            if len(plateau):
                peak = a + int(plateau[0])
            attack_end = min(b, int(np.searchsorted(t, t[peak] + 0.12)))
            finish_seen = t[b] - t[peak] >= 0.10 or (
                b + 2 < len(run) and stance[b + 2] < stance[peak] - 0.05
            )
            guard_before = (t < t[a]) & (t >= t[a] - 0.8) & (abs(stance - base["stance"]) < 0.08)
            complete = finish_seen and bool(np.any(guard_before))
            action = Action(
                "lunge",
                float(t[start]),
                float(t[attack_end]),
                float(t[peak]),
                complete,
                {
                    "trigger": "Stance expansion with front-knee flexion and asymmetric leg configuration.",
                    "recognizer": "temporal_rules_v1",
                },
            )
            run_actions.append(action)
            used[start : attack_end + 1] = True
            # Count the recovery once the fencer returns to guard and holds it briefly.
            returned = np.flatnonzero(
                (t > t[peak] + 0.18)
                & (abs(stance - base["stance"]) < 0.10)
                & (abs(front_knee - base["front_knee"]) < 20)
                & (abs(rear_knee - base["rear_knee"]) < 20)
            )
            returned = [int(i) for i in returned if t[i] - t[peak] <= 4]
            if returned:
                r = returned[0]
                end = int(np.searchsorted(t, t[r] + 0.18))
                if end < len(run) and np.all(abs(stance[r : end + 1] - base["stance"]) < 0.12):
                    recovery_candidates = np.flatnonzero(
                        (np.arange(len(t)) >= attack_end)
                        & (np.arange(len(t)) <= r)
                        & (stance < stance[peak] - 0.035)
                    )
                    recovery_start = (
                        max(attack_end, int(recovery_candidates[0]) - 1)
                        if len(recovery_candidates)
                        else r
                    )
                    run_actions.append(
                        Action(
                            "recovery",
                            float(t[recovery_start]),
                            float(t[end]),
                            float(t[end]),
                            True,
                            {
                                "trigger": "Observed return to the calibrated en-garde configuration."
                            },
                        )
                    )
                    used[attack_end : end + 1] = True
                else:
                    used[attack_end : r + 1] = True
            else:
                # Keep the held finish with the lunge until the stance narrows.
                hold_end = b + 1
                while hold_end < len(run) and stance[hold_end] > base["stance"] + 0.10:
                    hold_end += 1
                used[attack_end:hold_end] = True

        # The 3D coordinates move with the hips. For steps, use the video image
        # when the camera is fixed and forward or backward motion is visible.
        if calibration.movement_observable:
            fs, rs = v("front_screen"), v("rear_screen")
            vf, vr = velocity(fs, t), velocity(rs, t)
            movement = (~used) & ((abs(vf) > 0.18) | (abs(vr) > 0.18))
            for a, b in spans(movement, t, merge_gap=0.20):
                if t[b] - t[a] < config.min_action_seconds:
                    continue
                a = max(0, a - 1)
                b = min(len(run) - 1, b + 1)
                df, dr = fs[b] - fs[a], rs[b] - rs[a]
                if df > 0.10 and dr > 0.10:
                    kind = "advance"
                elif df < -0.10 and dr < -0.10:
                    kind = "retreat"
                else:
                    continue
                complete = b < len(run) - 1 and max(abs(vf[b]), abs(vr[b])) < 0.55
                run_actions.append(
                    Action(
                        kind,
                        float(t[a]),
                        float(t[b]),
                        float(t[b]),
                        complete,
                        {
                            "front_displacement_torso_units": float(df),
                            "rear_displacement_torso_units": float(dr),
                            "trigger": "Both feet translated in the same observable attack-axis direction.",
                        },
                    )
                )
                used[a : b + 1] = True

        # Check for the arm to extend while the distance between the feet stays steady.
        elbow, reach = v("elbow"), v("reach")
        arm_mask = (
            (~used)
            & np.isfinite(elbow)
            & (elbow > max(150, base.get("elbow", 120) + 15))
            & (reach > base.get("reach", 0.4) + 0.12)
            & (abs(stance - base["stance"]) < 0.12)
        )
        for a, b in spans(arm_mask, t, merge_gap=0.16):
            if t[b] - t[a] < config.min_action_seconds:
                continue
            start = a
            while (
                start > 0
                and not used[start - 1]
                and t[a] - t[start - 1] < 0.7
                and elbow[start - 1] > base.get("elbow", 120) + 4
            ):
                start -= 1
            start = max(0, start - 1)
            peak = a + int(np.nanargmax(elbow[a : b + 1]))
            run_actions.append(
                Action(
                    "arm_extension",
                    float(t[start]),
                    float(t[b]),
                    float(t[peak]),
                    b < len(run) - 1 or t[b] - t[peak] > 0.15,
                    {
                        "trigger": "Weapon-side elbow extension with forward reach and a stable stance.",
                        "claim_limit": "No blade direction, target contact, or hit is inferred.",
                    },
                )
            )
            used[start : b + 1] = True

        # Count a lunge and the advance just before it as one movement.
        run_actions.sort(key=lambda x: x.start)
        combined = []
        for action in run_actions:
            if (
                action.kind == "lunge"
                and combined
                and combined[-1].kind == "advance"
                and -0.08 <= action.start - combined[-1].end <= 0.65
            ):
                advance = combined.pop()
                action.kind = "advance_lunge"
                action.evidence["advance_interval"] = [advance.start, advance.end]
                action.evidence["lunge_start"] = action.start
                action.start = advance.start
                action.complete = action.complete and advance.complete
            combined.append(action)

        # Label en garde only when the fencer holds the guard.
        guard = (
            (~used)
            & (abs(stance - base["stance"]) < 0.10)
            & (front_knee >= 95)
            & (front_knee <= 160)
            & (rear_knee >= 95)
            & (rear_knee <= 160)
        )
        guard &= (
            (abs(velocity(v("hip_screen"), t)) < 0.12)
            & (abs(velocity(front_knee, t)) < 25)
            & (abs(velocity(rear_knee, t)) < 25)
        )
        if np.isfinite(elbow).any():
            guard &= ~np.isfinite(elbow) | (abs(velocity(elbow, t)) < 35)
        for a, b in spans(guard, t):
            if t[b] - t[a] >= 0.8:
                combined.append(
                    Action(
                        "en_garde",
                        float(t[a]),
                        float(t[b]),
                        float(t[(a + b) // 2]),
                        True,
                        {
                            "trigger": "Sustained calibrated guard configuration with low observed motion."
                        },
                    )
                )
                used[a : b + 1] = True
        unexplained = (~used) & (
            (abs(stance - base["stance"]) > 0.12)
            | (abs(velocity(front_knee, t)) > 35)
            | (abs(velocity(rear_knee, t)) > 35)
        )
        for a, b in spans(unexplained, t, merge_gap=0.12):
            if t[b] - t[a] >= 0.25:
                combined.append(
                    Action(
                        "unknown",
                        float(t[a]),
                        float(t[b]),
                        float(t[a]),
                        False,
                        {
                            "reason": "Available measurements do not uniquely match a supported movement rule."
                        },
                    )
                )
        actions.extend(combined)
    return sorted(actions, key=lambda x: (x.start, x.end))
