# Delivery verification - 2026-11-18

## Fresh execution evidence

The following checks were executed against this implementation. The test count describes software checks, not the number of athlete videos or a model accuracy percentage.

| Check | Observed result |
| --- | --- |
| Full suite with `FENCING_VISION_TESTS=1 python -m pytest -q` | **44 passed**, including the Streamlit application test and two real model/codec integration checks |
| Clean virtual-environment installation from `requirements.txt`, followed by editable package/dev installation | Successful |
| `python -m pip check` in that clean environment | No broken requirements found |
| Clean environment core/UI suite without optional integration flag | 42 passed, 2 intentionally skipped integration checks |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | Passed |
| `python -m compileall -q fencing_coach app.py tools` | Passed |
| `git diff --check` | Passed |
| Synthetic CLI -> PDF, JSON, CSV and landmark replay | Passed; all data are explicitly labelled simulated |
| Streamlit synthetic-demo workflow | Opened successfully, generated a PDF, displayed results, and exposed downloads without an application exception |
| Original sword checkpoint | Loaded and predicted with actual Ultralytics inference; task `segment`, class map `{0: "sword"}`; original SHA-256 unchanged |
| Blank-video end-to-end check | Actual decoder, pose model, reliability rejection, unscored PDF, and decodable annotated MP4 all exercised successfully |
| Public fencing-video smoke check | Actual MediaPipe and sword inference executed; the configured visibility gate rejected initial calibration and the system generated an unscored report rather than invented corrections |
| Official pinned pose-asset download helper | Downloaded model version 1; exact SHA-256 recorded in the manifest |
| Example PDF inspection | All five pages of the final side-view example rendered and visually checked; embedded fonts, evidence tables, coaching text and page labels are legible |

The inherited development environment emitted one unrelated `anyio`/pytest deprecation warning. It did not fail the suite. The clean-environment dependency check passed.

## Tested environment

- Python 3.12.14, Linux x86_64, CPU inference.
- NumPy 1.26.4.
- ReportLab 4.5.1.
- MediaPipe 0.10.21, using the Tasks API.
- OpenCV 4.11.0.86.
- Streamlit 1.49.1.
- PyArrow 20.0.0.
- Optional sword inference: Ultralytics 8.4.215 with CPU PyTorch 2.15.0 and torchvision 0.29.2.

The newer unpinned MediaPipe distribution encountered a native runtime failure in this environment. The tested implementation therefore fixes the MediaPipe version and compatible NumPy/OpenCV/Streamlit/PyArrow dependencies. The pinned official pose asset is included in the delivery.

## Explicitly unverified

- Physical webcam hardware and operating-system camera permission: no actual camera device was available for testing. The route is implemented, but this hardware outcome is not claimed as tested.
- Windows batch execution: reviewed and provided for Python 3.11/3.12, but not executed on Windows in this Linux environment.
- Real-world action recognition precision/recall across athletes and camera angles: no exhaustively action-labelled benchmark was supplied.
- Agreement between /100 prototype scores and independent fencing coaches: no coach-scored dataset was supplied.
- Newly trained action models or retrained sword weights: none are claimed or silently substituted.

The public smoke clip was reached through MIT OpenCourseWare PE.740's individual advance reference: https://ocw.mit.edu/courses/pe-740-fencing-spring-2007/pages/video/ . Its external video bytes are not redistributed in this package. This was an integration/rejection check, not a labelled action-accuracy benchmark.
