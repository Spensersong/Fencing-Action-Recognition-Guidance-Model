# Testing and evaluation

## Run the tests

Install the app requirements first if you want to exercise the Streamlit interface.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

The suite checks action rules, both handedness settings, rotated 3D coordinates, missing landmarks, incomplete movements, score calculations, and PDF and data exports. The coordinate tests use simulated poses with known geometry.

Install `requirements-sword.txt` to include the sword model test. Then run the vision checks with the pose model present.

```bash
FENCING_VISION_TESTS=1 python -m pytest -q
```

In PowerShell, set the flag with `$env:FENCING_VISION_TESTS = "1"`, then run `python -m pytest -q`. The vision tests check model loading, decoding, video export and rejection of a clip without a visible athlete.

## Evaluate on recorded drills

The reference ranges and action recognition rules are awaiting a benchmark with independently labelled fencing videos. To evaluate them, record consented clips of different athletes from side, angled and front views. Keep all views of the same performance together and split development and evaluation sets by athlete and recording session.

Fill in [the label template](../validation/labels_template.csv). Mark each supported movement interval in a clip, including any en garde holds. If a coach score is available, use the same rubric for each assessment in the benchmark.

| Column | Value |
| --- | --- |
| `video_id` | The `source_name` in the report JSON, usually the clip filename. |
| `view` | The recorded camera view. Use clips with one view each. |
| `action` | `en_garde`, `advance`, `retreat`, `arm_extension`, `lunge`, `recovery` or `advance_lunge`. |
| `start`, `end` | Start and end in seconds, with the end after the start. |
| `coach_score` | Optional score from 0 to 100 for this interval. |

Save each clip's analysis in its own subfolder, then run the evaluator on the parent directory.

```bash
python tools/evaluate.py --labels validation/labels.csv --predictions outputs/held_out --output outputs/benchmark.json --iou 0.5
```

The evaluator matches intervals with the same label and sufficient time overlap. A rejected clip still counts toward recall through its labelled intervals. The output includes precision, recall, F1, results by view and action, and score coverage. When a match has both a program score and a coach score, the output also reports the paired count and mean absolute error. Read score error alongside coverage and recall to see how many clips had usable scores.
