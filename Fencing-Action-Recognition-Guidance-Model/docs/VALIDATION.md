# Verification and validation

## Distinguish three kinds of evidence

| Layer | Appropriate evidence | What it cannot establish |
| --- | --- | --- |
| Software correctness | Unit, integration, export, UI, and failure-path tests | Real-world recognition accuracy |
| Geometric behaviour | Specified analytical joint coordinates under rotation/scale/handedness changes | MediaPipe's camera-dependent landmark error |
| Sports assessment validity | Independent athlete videos, exhaustive action labels, blinded coach ratings | Nothing of this kind can be inferred from test counts alone |

The current project has executable checks for the first two layers. `docs/VERIFICATION_RESULTS.md` records the fresh delivery checks. No benchmark percentage for real fencing action recognition is asserted.

## Existing automated checks

Checks cover supported drill names, classical front/rear leg asymmetry, invariant 3D angle calculations under yaw, handedness, non-square frames, frontal-view abstention, unavailable weapon-arm landmarks, interrupted observations, multiple people, missing poses, moving cameras, incomplete finishes, measured trunk deviations, timestamp errors, invalid geometry, rubric weights, range calculations, English report generation, finite JSON, and landmark replay.

UI checks use Streamlit's actual application test harness. Optional video checks use real OpenCV decoding, the official MediaPipe model, local PDF export, and video encoding. These do not replace a labelled action dataset. A rejected real-video capture is a successful failure-path test, not proof of good action classification.

The physical webcam route is implemented but cannot be confirmed without access to an actual camera device. Acquisition timestamps are monotonic wall-clock times. Slow local inference can leave gaps; such gaps are reported and are not repaired by inventing observations.

## Build a held-out benchmark

Use consented clips with stable timestamps and original playback speed. Include left/right-handed athletes, varied experience and body proportions, side and oblique views, strongly frontal views, varied lighting, occlusions, incomplete movements, camera movement, and unsupported actions. Record view labels independently of the program's estimated view bins.

Split **by athlete and recording session**, not by neighbouring frames. Do not place rotations, mirrored copies, or other transformations of the same source clip in both development and final testing. If several camera views show the same performance, keep that performance together in a single split. Reserve final evaluation before tuning rules.

Have coaches define the drill and score each dimension using an agreed rubric. Measure inter-rater agreement and keep disagreements. Scores from different weapons or tactical tasks are not automatically interchangeable.

## Label schema

The CSV needs `video_id,view,action,start,end,coach_score`:

- `video_id`: must exactly match the `source_name` in the corresponding `analysis.json` (normally the uploaded video's basename).
- `view`: independently recorded capture view, such as `side`, `oblique`, or `frontal`.
- `action`: one of `en_garde`, `advance`, `retreat`, `arm_extension`, `lunge`, `recovery`, `advance_lunge`.
- `start`, `end`: seconds in the original video, with `end > start`.
- `coach_score`: optional agreed 0-100 score for that exact drill and rubric.

Exhaustively annotate all supported action intervals, including static guard intervals that the program is allowed to label. If only one movement is annotated while other movements are ignored, predicted unannotated movements would be counted as false positives incorrectly. Use a separate `video_id` for a changing-view segment. The current CSV evaluator is intended for one consistent view per clip.

Run analysis into separate directories under `outputs/held_out/`, then:

```bash
python tools/evaluate.py --labels validation/labels.csv --predictions outputs/held_out --output outputs/benchmark.json --iou 0.5
```

The evaluator pairs same-label intervals with temporal IoU at least 0.5 using a greedy descending-overlap match. One prediction can match at most one truth interval. Wrong labels contribute a false positive and false negative. A missing analysis output or rejected capture contributes false negatives for its annotated actions. Unclassified intervals do not masquerade as correct labels. Report the IoU threshold with every result.

Outputs include action precision, recall, F1, per-view/per-action groups, matched-action score coverage, paired coach-score count, and MAE. Undefined quantities remain null. Synthetic result files are refused. Score MAE alone is insufficient: it must be read alongside scoring coverage, recognition recall, and sample counts. Report per-athlete uncertainty when a suitable sample is available; this helper does not manufacture statistical confidence intervals for a small dataset.

## Training status

There was no action-labelled or coach-scored dataset in the supplied repository. No action classifier has been trained or retrained, and no new sword segmentation accuracy has been measured. The `.pt` weights are the original user's weights. The application does not call its hand-authored scoring rules a trained professional coaching model.
