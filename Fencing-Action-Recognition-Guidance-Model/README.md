# Fencing Evidence Coach

A Python application for reviewing solo fencing drills with **MediaPipe 3D pose estimates, temporal movement rules, an optional YOLO sword segmentation model, and evidence-linked English PDF reports**.

Upload a video in the local Streamlit application, capture a drill with a webcam connected to the Python computer, or run the command-line tool. Analysis runs locally. No generative language model writes the observations or coaching cues.

**Status:** working research prototype, not a validated fencing grading system. Every numerical tolerance is disclosed. Unsupported observations remain unassessed. A numerical score describes agreement with the configured reference drill; it does not certify fencing ability.

## Quick start on Windows

Use **64-bit Python 3.11 or 3.12**. Open a terminal in this repository:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively, double-click `start_windows.bat`. Its first run creates a virtual environment and installs the application dependencies. Subsequent runs reuse that environment.

The delivery package includes `models/pose_landmarker_full.task`. If that file was omitted when downloading the repository, retrieve the official asset:

```powershell
.venv\Scripts\python.exe tools/download_pose_model.py
```

To enable the existing sword model:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-sword.txt
```

Then check **Include my sword segmentation model** in the application. The default path is the actual repository filename, `best(1).pt`.

## Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

On a minimal Ubuntu/Debian installation, MediaPipe/OpenCV may also require the operating-system packages `libegl1`, `libgles2`, `libgl1`, and `libglib2.0-0`. A GPU is not required. The tested CPU implementation uses MediaPipe **0.10.21** and its **Tasks API**; it does not depend on the removed `mp.solutions` interface in newer releases. Keep the provided compatible NumPy, OpenCV, Streamlit, and PyArrow constraints when installing.

## Recording a usable drill

1. Keep the camera fixed and level. Use a side or oblique view that shows **both knees, ankles, hips, and shoulders**.
2. Keep one athlete in frame, including both feet and the weapon-side arm. Avoid loose clothing that hides the joints.
3. Begin with **at least 1.2 seconds of still en garde**. This defines the front-to-rear foot axis, body scale, and reference guard.
4. Select the athlete's actual weapon hand and front leg. If the saved input is mirrored, enable **The source video is mirrored**.
5. Perform a supported solo drill and hold a clear finish. Record at 30 FPS or higher when possible. A short clip with one movement is easiest to review.

The application asks whether the camera is fixed and level because a single skeleton does not establish camera motion reliably. It does not claim to prove that a user-supplied capture declaration is true. Frontal video cannot support every translation-dependent measurement. A small, occluded, or poorly framed athlete can cause calibration to be rejected; this is reported without issuing movement corrections.

## Supported movement rules

| Movement | Observable temporal evidence | Important limit |
| --- | --- | --- |
| En garde | Sustained guard configuration with low observed motion | A supported hold, not a label assigned to every idle interval |
| Advance | Both feet translate along the calibrated forward image direction | Requires fixed side/oblique capture |
| Retreat | Both feet translate in the opposite direction | Requires fixed side/oblique capture |
| Straight arm extension | Weapon-side elbow extends and reach increases with stable foot spacing | Does not prove a straight blade, a thrust landing, or target contact |
| Lunge | Foot-spacing expansion, front-knee flexion, and asymmetric leg configuration over time | Only the supported solo-drill pattern; not every attack style |
| Recovery | An observed return from lunge to a stable calibrated guard | No completed recovery is invented after a cutoff |
| Advance-lunge | Observed advance immediately followed by a lunge | Both components retained; their scores receive equal weight |
| Unclassified | Motion does not match a supported rule uniquely | No action-specific score or correction |

This is a **rule-based action recognizer**, not an action classifier trained on the supplied `.pt` file. The sword checkpoint is a YOLOv8n segmentation model with one class, `sword`. It provides sword-region polygons and boxes. It contains no separate sword-tip, guard, or handle labels. See `models/manifest.json` and `tools/inspect_checkpoint.py`.

## Scoring and PDF guidance

| Dimension | Maximum | Examples of measured evidence |
| --- | ---: | --- |
| Joint configuration | 45 | 3D knee/elbow angles, trunk inclination, final stance configuration |
| Timing and coordination | 25 | Separately observed arm/foot onsets, foot order, drill duration |
| Movement control | 20 | Supported foot-spacing and image-displacement measures |
| Fluency and consistency | 10 | Trunk-angle variation and shoulder-relative reach variation |

The fluency category replaces subjective appearance ratings with explicit motion proxies. It does not score physique, beauty, or stylistic preference.

Every PDF contains movement intervals, measured values, timestamps, reference bands, point contributions, fixed coaching review cues, missing-evidence reasons, and source/method information. Full scores are withheld if **any weighted metric** or the action finish is unobserved. Missing weights are never silently redistributed to available metrics.

All numerical bands, weights, and fixed cues are in [`fencing_coach/rubric.json`](fencing_coach/rubric.json). They are **project-defined prototype settings**, not federation numerical standards. Classical lunge descriptions and MIT demonstration categories inform selected cues. Timing, posture, and style vary by drill, weapon, athlete, and tactical purpose; the program does not treat one reference drill as a universal competitive technique.

See [`docs/METHOD.md`](docs/METHOD.md) for equations and abstention rules, [`docs/VALIDATION.md`](docs/VALIDATION.md) for evidence limits and evaluation, and [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) for the role of each component.

## Command-line examples

```bash
# Video and PDF, without the optional sword dependency
python -m fencing_coach.cli --video my_lunge.mp4 --output outputs/my_lunge

# Video with the original sword segmentation weights
python -m fencing_coach.cli --video my_lunge.mp4 --sword-model "best(1).pt" --output outputs/sword_session

# Left-handed athlete
python -m fencing_coach.cli --video left_handed.mp4 --weapon-hand left --front-leg left --output outputs/left

# Local webcam: use acquisition timestamps and analyse the full capture
python -m fencing_coach.cli --camera 0 --seconds 15 --output outputs/webcam

# Explicitly mark a moving camera; translation-dependent claims are withheld
python -m fencing_coach.cli --video moving_camera.mp4 --moving-camera --output outputs/moving

# Reproducible analytical demo, with no vision dependencies or camera
python -m fencing_coach.cli --demo --output outputs/demo

# Replay exported landmarks without running the pose model again
python -m fencing_coach.cli --landmarks outputs/my_lunge/landmarks.json --output outputs/replay
```

Each session writes:

| File | Purpose |
| --- | --- |
| `coaching_report.pdf` | English, evidence-linked guidance |
| `analysis.json` | Structured action, metric, scoring, limitation, and provenance record |
| `measurements.csv` | Timestamped feature measurements and exclusions |
| `landmarks.json` | Estimated image/world landmarks for repeatable analysis |
| `annotated_video.mp4` | Optional pose/segmentation overlays and supported action labels; no audio |

`landmarks.json` contains model estimates, not laboratory motion-capture ground truth. The annotated MP4 uses the local OpenCV encoder; browser codec support varies, but it can be downloaded with the evidence bundle.

## Testing and real-world evaluation

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m compileall -q fencing_coach app.py tools
```

Synthetic fixtures verify specified kinematics, coordinate rotation, handedness, aspect ratios, timing, missing evidence, and report behaviour. They do **not** establish real-world recognition accuracy or scoring validity. The included report in `examples/` is clearly marked as a synthetic software demonstration.

For a real benchmark, exhaustively annotate independent athlete videos with the schema in `validation/labels_template.csv`, then run:

```bash
python tools/evaluate.py --labels validation/labels.csv --predictions outputs/held_out --output outputs/validation.json
```

The evaluator reports interval precision/recall/F1, per-view results, score coverage, and paired coach-score MAE. Rejected videos remain in the recall denominator. Synthetic results are rejected by this evaluation route. No action-model retraining, held-out accuracy percentage, or professional coach agreement is claimed for this delivery because an appropriate labelled action dataset was not supplied.

## Uploading this version to GitHub

The delivery ZIP contains the application source, both inference assets, tests, English documentation, and sample reports. Extract the **contents** into your repository, replacing the old application and README. Keep `best(1).pt` unchanged. The previous `final fencing version` text program is superseded; its original version remains in your Git history. Do not upload `.venv/`, caches, personal training videos, or `outputs/`; those paths are ignored.

This delivery does not push changes to your GitHub account. Review [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), particularly the existing Ultralytics/checkpoint license metadata, before redistributing the model or choosing a project license.
