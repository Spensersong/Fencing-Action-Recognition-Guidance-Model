# Measurement and scoring method

## What the system can establish

This application measures agreement with explicitly configured **beginner solo-drill references**. It does not infer intention, opponent distance, an actual touch, a referee decision, blade contact, medical status, or subjective beauty.

There are two different kinds of machine learning in the workflow:

1. The pretrained MediaPipe model estimates human landmarks from pixels.
2. The user's existing YOLO checkpoint estimates a sword segmentation region.

The new action recognizer and report writer use deterministic temporal and scoring rules. They are not a newly trained fencing action network and do not call a language model. This removes free-form invented report content, **not all errors in image-based pose estimation**.

## Landmark contract

Each timestamp has two `(33, 5)` arrays: `[x, y, z, visibility, presence]`.

| Array | Coordinates | Use |
| --- | --- | --- |
| `image` | x normalized by image width; y by height; model-relative z | Crop/visibility checks, properly scaled image displacement, overlays |
| `world` | Estimated 3D metres, origin at the midpoint of the hips | Joint angles, body-relative directions, limb proportions |

Core indices are shoulders 11/12, hips 23/24, knees 25/26, ankles 27/28. The weapon-side shoulder/elbow/wrist triplets are 11/13/15 for left and 12/14/16 for right.

**Important:** hip-relative world coordinates cannot measure whole-body translation. The hip midpoint stays near the origin even when a person walks. Advance/retreat detection therefore uses image displacement only when the attack direction has sufficient image projection and the camera is declared fixed and level.

Normalized image x and y do not share the same physical scale. The implementation converts them to pixels by multiplying x by width and y by height before computing image-space distances or directions. It never computes joint angles directly from the uncorrected normalized 2D coordinates.

## Calibration

The first 1.2 seconds must contain a stable en-garde hold. Required body landmarks must satisfy visibility/presence and crop checks. Default per-landmark threshold: 0.65. At least five usable observations spanning 0.45 seconds are required. Coverage below 80%, an observation gap greater than 0.20 seconds, moving calibration, implausible limb lengths, or unsupported guard geometry rejects calibration.

The initial front ankle minus rear ankle, with its vertical component removed, defines the attack axis. Its median direction is fixed for the session. Leg length is the mean of the two thigh-plus-shank lengths. Image distance is normalized by the initial shoulder-midpoint-to-hip-midpoint pixel distance. These are two different units: `leg` is estimated 3D leg length; `torso` is initial projected torso length.

The initial knee-angle acceptance bands (90-165 degrees) and stance acceptance band (0.25-1.45 leg lengths) are operational prototype bounds, not a definition of every valid en-garde style. If they do not fit a coach-approved training style, a separate validated reference profile is needed; the software must not quietly label that style incorrect.

The attack-axis horizontal image-projection ratio is estimated from its 3D camera-x component:

- At least 0.82: `side`.
- At least 0.45 and below 0.82: `oblique`.
- Below 0.45: `frontal` or strongly depth-oriented.

These are observability bins, not calibrated camera-angle measurements. They depend on the pose model's depth estimate. Camera orientation is not solved by a calibrated multiview reconstruction.

## Geometry

For joint points A-B-C:

`angle = arccos(dot(A-B, C-B) / (norm(A-B) * norm(C-B)))`.

Zero-length or non-finite vectors return no measurement. Trunk inclination is the angle between the hip-to-shoulder midpoint vector and the assumed upright camera-y direction. It therefore requires a level camera. A fixed attack axis, 3D angles, explicit handedness, and scale normalization improve geometric consistency under view changes; they do not recover anatomy hidden by occlusion.

Front and rear knees have **different** classical lunge roles. No penalty is applied simply because the left and right knee angles differ.

## Reliability gates

The program rejects a sampled frame when core landmarks are unavailable, outside the image tolerance, non-finite, or below confidence thresholds; the detector reports multiple people; limb lengths deviate more than 35% from calibration; the foot axis turns more than 32 degrees from calibration; or an abrupt image jump exceeds the configured threshold.

The confidence numbers are model scores, not calibrated probabilities of correctness. A high-confidence, stable pose can still be wrong. The program does not claim to detect every camera pan, cut, mirror error, joint swap, or image ambiguity. Whole-session fixed/level declarations remain capture assumptions. A supplied mirrored source is unmirrored before inference.

Missing observations break temporal runs. The code does not forward-fill a missing angle or interpolate across a gap. Feature smoothing is a local median within 0.065 seconds on either side of each observation, inside a reliable run only. Event order is withheld unless sustained onsets differ by more than `max(0.07 s, 2 * median sample interval)`. Values at or below this resolution must not be interpreted as resolved sequencing.

## Temporal recognition

The recognizer identifies expanded stance/front-knee flexion plus asymmetric leg configuration for lunges; a measured return to the initial guard for recovery; matched-direction translation of both feet for steps; and weapon-arm extension with stable foot spacing for an arm-only drill. An advance immediately followed by a lunge is combined without double-counting its components.

Recognition is intentionally separate from the quality rubric: a non-ideal knee angle can still be identified and measured. These hand-authored rules support the documented drills; they have no demonstrated coverage of all valid competitive attacks, running actions, jumps, fleches, parries, or tactical variations.

When an interval does not match a supported pattern, it receives `unknown` with no action-specific correction. A cut-off finish withholds an overall score. No recovery, foot contact, or blade touch is inferred beyond the observed interval.

## Metric definitions

| Metric | Definition | Limit |
| --- | --- | --- |
| Knee/elbow angle | Median 3D angle near the action peak or final window | Estimated anatomy, not a laboratory measurement |
| Trunk inclination | 3D hip-to-shoulder midpoint angle from camera vertical | Requires level capture |
| Stance ratio | Ankle separation along the initial attack axis / estimated leg length | Relative body geometry, not piste distance |
| Stance error | Absolute final stance-ratio difference from calibrated guard | A reference comparison, not a universal ideal |
| Knee offset | Absolute knee-to-ankle displacement along attack axis / leg length | Does not measure knee load or diagnose an unsafe posture |
| Arm lead | Front-foot forward onset time minus arm-extension onset time | No priority determination; unresolved onsets are ungraded |
| Foot order | Following-foot onset minus initiating-foot onset for the selected step | Requires supported projection and separate onsets |
| Movement duration | Observed onset-to-peak or segment duration as labelled | Not reaction time; capture playback speed must be correct |
| Hip excursion | Range of hip-midpoint image-y / initial projected torso length | Not centre of mass, actual balance, or a calibrated vertical distance |
| Trunk variation | Standard deviation of observed trunk angle over the interval | Consistency proxy; intentional movement can be appropriate |
| Reach-path ratio | Total variation of shoulder-relative axial reach / its range | Not sword-tip path, smoothness ground truth, or target accuracy |

Peak values are local-window medians, not an assertion that the displayed video frame has an exact ground-truth joint angle. Aggregate metrics describe the whole marked interval; their recorded timestamp is an evidence marker or interval endpoint. Displayed decimal precision is for reproducibility and does not imply corresponding measurement accuracy.

## The transparent /100 rubric

`fencing_coach/rubric.json` is authoritative. Each metric lists a label, dimension, original weight, unit, acceptable reference band, outer zero-score bounds, fixed coaching cue, and source category.

Inside the reference band, the metric score is 100. Outside it, the score decreases linearly toward zero at its outer bound, and is clipped to 0-100. Weighted points equal `metric_score * metric_weight / 100`.

| Dimension | Original weight |
| --- | ---: |
| Joint configuration | 45 |
| Timing and coordination | 25 |
| Movement control | 20 |
| Fluency and consistency | 10 |

A dimension is displayed only when all its metrics are observed. An overall /100 score is displayed only when **all** weighted metrics and a completed finish are observed. Unknown weights are not redistributed. Weighted evidence coverage is the sum of supported metric weights divided by 100. Data visibility is reported separately and is never substituted for athletic quality.

For an advance-lunge, the advance and lunge each contribute half of each dimension's original weight. If either component lacks a required measurement, the compound overall score is withheld.

The report writer does not infer a cause from a deviation. It states the measurement, reference band, time/interval, and a fixed coaching review cue. It never invents weakness, fatigue, poor intent, joint loading, or tactical failure. A deviation from a project reference does not establish that an athlete's actual technique is inappropriate.

## Sources and what they do not establish

1. FIE-hosted *Lunge performance and its determinants* describes a classical lunge, including an extending weapon arm, front-leg flexion, rear-leg extension, and relatively upright torso: https://static.fie.org/uploads/24/124021-LUNGE%20PERFORMANCE%20AND%20ITS%20DETERMINANTS.pdf
2. MIT OpenCourseWare PE.740 provides basic movement demonstration categories: https://ocw.mit.edu/courses/pe-740-fencing-spring-2007/pages/video/
3. Google's MediaPipe documentation defines pose landmarks, world/image coordinate outputs, and video timestamps: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
4. Ultralytics documents segmentation inference outputs: https://docs.ultralytics.com/modes/predict/
5. Label Studio documents YOLO annotation/model workflows: https://labelstud.io/guide/ml_tutorials/yolo

These sources do not establish this project's numerical angle bands, timing cutoffs, scoring weights, accuracy, or coach agreement. Those remain project choices awaiting independent validation.
