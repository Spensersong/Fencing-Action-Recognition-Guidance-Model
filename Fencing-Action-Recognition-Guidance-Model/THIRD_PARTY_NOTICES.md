# Third-party assets and references

## Original sword segmentation checkpoint

- File: `best(1).pt` (original bytes preserved).
- Origin: the user's `Spensersong/Fencing-Action-Recognition-Guidance-Model` repository.
- Checkpoint metadata: Ultralytics version `8.4.60`; task `segment`; architecture `yolov8n-seg`; one class, `sword`.
- Checkpoint-declared license string: `AGPL-3.0 (https://ultralytics.com/license)`.
- The runtime adapter uses the Ultralytics library. No claim is made about the provenance or redistribution rights of the original training images; they were not supplied.
- The manifest records the exact checksum. No action-recognition training was performed on this checkpoint.

## MediaPipe Full Pose Landmarker

- File: `models/pose_landmarker_full.task`.
- Official asset: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
- Model documentation/card: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- MediaPipe project: https://github.com/google-ai-edge/mediapipe (Apache-2.0 project license; consult the model card and asset terms for the model).
- This package does not claim ownership of the pretrained pose model. The model manifest and download helper preserve its origin and checksum.

## Coaching references

- FIE-hosted *Lunge performance and its determinants*: https://static.fie.org/uploads/24/124021-LUNGE%20PERFORMANCE%20AND%20ITS%20DETERMINANTS.pdf
- MIT OpenCourseWare PE.740, Fencing, Spring 2007, movement videos: https://ocw.mit.edu/courses/pe-740-fencing-spring-2007/pages/video/

Descriptions and cues in this project are paraphrased. No reference PDF, external training video, or human-authored textbook passage is redistributed in the delivery. Numerical scoring tolerances are project-defined and must not be attributed to these sources as official standards.

## Python dependencies

Third-party Python libraries retain their own licenses, including NumPy, ReportLab, OpenCV, MediaPipe, Streamlit, PyArrow, PyTorch, and Ultralytics where installed. This notice does not relicense the original project, its dependencies, or its checkpoint. No new blanket project license has been selected on the user's behalf.
