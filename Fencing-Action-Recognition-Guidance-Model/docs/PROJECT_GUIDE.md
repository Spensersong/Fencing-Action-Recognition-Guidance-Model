# Project guide

## What your existing file actually is

`best(1).pt` is a **trained PyTorch/Ultralytics checkpoint**, not the raw dataset and not the full application. Static metadata inspection identifies a YOLOv8n segmentation model trained with Ultralytics 8.4.60. Loading it with the runtime confirms the task `segment` and class map `{0: "sword"}`. Its exact original checksum is retained in `models/manifest.json`.

Label Studio is the annotation tool. It can be used to mark image regions and export training annotations. A training workflow consumes the annotated images and produces the `.pt` weights. The checkpoint alone does not provide the original images, annotations, independent validation split, segmentation accuracy, or detailed sword-part labels.

The current one-class mask cannot establish sword-tip position, which end is the tip, grip quality, parry type, blade contact, point control, target validity, or successful touch. The application shows the observed mask/box and confidence without inventing those additional labels.

## How the new pipeline works

1. Decode uploaded video or acquire timestamped webcam frames.
2. Estimate 33 body landmarks using the official MediaPipe Full Pose Landmarker model.
3. Optionally run your existing sword segmentation checkpoint on the same frames.
4. Check visibility, crop, subject count, scale, timing continuity, and calibration.
5. Compute 3D joint geometry and observable image-space movement.
6. Segment supported solo drills over time.
7. Compare supported measurements with the disclosed reference rubric.
8. Write a fixed-template English PDF and machine-readable evidence files.

The sword model and body analysis are independent: absence of a detected sword does not stop a footwork drill, and a detected sword does not prove an attack occurred.

## File map

| File | Responsibility |
| --- | --- |
| `app.py` | English upload/webcam interface and downloads |
| `fencing_coach/schema.py` | Configuration, timestamped landmarks, action and result contracts |
| `fencing_coach/vision.py` | MediaPipe Tasks, trusted local YOLO weights, overlays |
| `fencing_coach/features.py` | Calibration, observability, rejection, local smoothing |
| `fencing_coach/geometry.py` | 3D angles and correctly scaled image geometry |
| `fencing_coach/recognition.py` | Temporal action segmentation rules |
| `fencing_coach/scoring.py` | Explicit measurements, coverage and original-weight grading |
| `fencing_coach/rubric.json` | Reference bands, weights, source categories and fixed cues |
| `fencing_coach/report.py` | English PDF layout and evidence-linked guidance |
| `fencing_coach/pipeline.py` | Video/webcam/replay workflows and export |
| `fencing_coach/demo.py` | Analytical synthetic test fixtures, not training data |
| `fencing_coach/evaluation.py` | Held-out interval and coach-score evaluation |
| `tools/` | Asset download, static checkpoint inspection, benchmark command |
| `tests/` | Behavioural checks and optional integration tests |

## How to make the system more professional

The next evidence needed is a real, consented, coach-labelled set of these specific drills. Changing the prose to sound authoritative would not make the measurements or reference ranges more valid. A useful progression is:

1. Have a qualified fencing coach approve each supported drill definition, weapon applicability, and correction cue.
2. Record multiple athletes and cameras, with repeated side/oblique/frontal views and deliberately included occlusion/poor-technique cases.
3. Mark action starts/ends and score dimensions independently of the program. Keep athlete identities separate across development and final testing.
4. Use `tools/evaluate.py` to measure per-view recognition, rejected-action coverage, and paired score error. Retain failed captures in the evaluation.
5. Adjust references only on development data, version the rubric, then measure again on untouched test athletes.
6. If rule coverage remains insufficient, train an explicit temporal classifier using those labelled pose sequences. Do not train a new fencing classifier from unlabelled videos or synthetic-only fixtures and present it as validated.

No new action training is claimed in this delivery. The actual supplied sword weights are preserved, loaded, and integrated. The repository did not contain the original sword training dataset, action-labelled videos, or coach ratings needed for defensible retraining or final accuracy claims.

## Multi-view boundary

The software uses 3D joint angles and an initial body-relative attack axis, rather than fixed 2D pixels, for posture. Its synthetic rotation tests cover the mathematical invariance of those calculations. A single camera still cannot provide reliable information about a fully hidden joint or guarantee depth accuracy. Strongly frontal captures deliberately withhold forward/backward translation and projection-dependent scores.

For full assessment of a drill with hidden joints, record a second clear view and analyse it separately. The program does not claim to synchronize or fuse multiple cameras. Calibrated multi-camera reconstruction would require camera calibration, synchronization, subject association, and a separate validation dataset.
