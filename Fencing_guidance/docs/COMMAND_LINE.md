# Command-line guide

Run these commands from the project folder after installing the packages listed in the [README](../README.md).

## Analyse a video

```bash
python -m fencing_coach.cli --video my_drill.mp4 --output outputs/my_drill
```

The defaults are foil, right weapon hand, right front leg and a fixed, level camera. Set the weapon and sides to match the recording.

```bash
python -m fencing_coach.cli --video left_drill.mp4 --weapon epee --weapon-hand left --front-leg left --output outputs/left_drill
```

| Option | When to use it |
| --- | --- |
| `--mirrored-input` | The saved clip is mirrored. |
| `--moving-camera` | The camera moves during the clip. The app leaves image-displacement measures unassessed. |
| `--camera-not-level` | The camera is tilted. Calibration requires a level camera. |
| `--sample-fps 20` | Limit pose analysis to 20 sampled frames per second. The default limit is 30. |
| `--max-seconds 60` | Analyse at most the first 60 seconds. The default limit is 300. |
| `--no-annotated-video` | Export the report and data only. |
| `--sword-model "best(1).pt"` | Include the sword outline after installing `requirements-sword.txt`. Use trusted local weights. |

Run `python -m fencing_coach.cli --help` to see all options.

## Record with a webcam

```bash
python -m fencing_coach.cli --camera 0 --seconds 15 --output outputs/webcam
```

The number after `--camera` is the local device ID. Start in en garde as soon as the recording begins. The program saves the PDF and measurements after the capture.

## Try a sample or reuse landmarks

```bash
python -m fencing_coach.cli --demo --output outputs/demo
python -m fencing_coach.cli --landmarks outputs/my_drill/landmarks.json --output outputs/replay
```

The demo uses simulated joint positions. Landmark replay uses the saved coordinates. Pass the same weapon, sides and camera settings again to review the same drill under the same settings.

## Output files

| File | Contents |
| --- | --- |
| `coaching_report.pdf` | Movement intervals, scores, measurements and suggestions for reviewing the recording. |
| `analysis.json` | Detected movements, reference ranges, points, missing measurements and recording settings. |
| `measurements.csv` | Per-frame measurements and reasons for excluding frames. |
| `landmarks.json` | The estimated body coordinates used for analysis. This file can be used for replay. |
| `annotated_video.mp4` | An optional overlay on an uploaded video. The file has no audio. |

If the app cannot find the pose model, restore it from the project or run `python tools/download_pose_model.py`. If OpenCV cannot read a clip, try a constant-frame-rate H.264 MP4 at the original playback speed.
