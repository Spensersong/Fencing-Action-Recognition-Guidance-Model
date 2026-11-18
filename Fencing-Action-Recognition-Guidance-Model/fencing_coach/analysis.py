"""Public in-memory analysis API, also used for reproducible landmark tests."""

from dataclasses import asdict

import numpy as np

from . import __version__
from .features import calibrate, extract_features
from .recognition import recognize
from .schema import AnalysisConfig, AnalysisResult
from .scoring import load_rubric, score_action


def analyze_frames(
    frames, config=None, source_name="Landmark sequence", simulated=False, provenance=None
):
    config = config or AnalysisConfig()
    frames = list(frames)
    if any(not np.isfinite(f.t) for f in frames) or any(
        b.t <= a.t for a, b in zip(frames, frames[1:], strict=False)
    ):
        raise ValueError("Frame timestamps must be finite and strictly increasing.")
    rubric = load_rubric()
    calibration = calibrate(frames, config)
    features = extract_features(frames, calibration, config)
    actions = recognize(features, calibration, config)
    results = [score_action(a, features, calibration, config, rubric) for a in actions]
    warnings = []
    if not calibration.valid:
        warnings.append(calibration.reason)
    elif not calibration.movement_observable:
        warnings.append(
            "Axial image motion is not sufficiently observable. Advance/retreat labels and projection-dependent scores are withheld."
        )
    if not config.camera_static:
        warnings.append(
            "The camera is marked as moving. Image displacement cannot establish athlete translation."
        )
    invalid = sum(not f.valid for f in features)
    if invalid:
        warnings.append(
            f"{invalid} of {len(features)} sampled frames were excluded from movement analysis."
        )
    warnings.extend(sorted({r for f in features if not f.valid for r in f.reasons})[:8])
    if any(
        b.t - a.t > config.max_gap_seconds + 1e-6 for a, b in zip(frames, frames[1:], strict=False)
    ):
        warnings.append(
            "Timestamp gaps were detected. Actions are never joined across an observation gap."
        )
    warnings.append(
        "Rules and scoring ranges have not been validated against a held-out coach-labelled fencing dataset. No recognition-accuracy claim is made."
    )
    if simulated:
        warnings.insert(
            0,
            "SIMULATED LANDMARKS: this demonstrates software behaviour, not a real athlete assessment.",
        )
    result = AnalysisResult(
        "1.0",
        __version__,
        rubric["version"],
        source_name,
        simulated,
        asdict(config),
        calibration,
        frames[-1].t - frames[0].t if len(frames) > 1 else 0,
        sum(f.valid for f in features) / len(features) if features else 0,
        results,
        list(dict.fromkeys(warnings)),
        provenance or {},
    )
    return result, features
