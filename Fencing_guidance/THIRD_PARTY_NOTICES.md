# Models and references

## Sword segmentation

`best(1).pt` contains YOLOv8n segmentation weights trained with Ultralytics 8.4.60 on fencing sword annotations made in Label Studio. Its class list contains one label, `sword`. The checkpoint records the Ultralytics AGPL-3.0 license notice. The training images and annotations are external to the repository. [Ultralytics licensing](https://www.ultralytics.com/license) · [Label Studio YOLO guide](https://labelstud.io/guide/ml_tutorials/yolo)

## Pose estimation

`models/pose_landmarker_full.task` is version 1 of Google's MediaPipe Full Pose Landmarker, float16. The model supplies the body landmarks used by the analysis. [MediaPipe project and license](https://github.com/google-ai-edge/mediapipe) · [Pose Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) · [Model download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)

The model files can be checked against [models/manifest.json](models/manifest.json). Python dependencies retain their respective licenses.

## Fencing references

- The description of a classical lunge informs the review cues for the front and rear legs, torso and weapon arm. [Lunge performance and its determinants, hosted by the FIE](https://static.fie.org/uploads/24/124021-LUNGE%20PERFORMANCE%20AND%20ITS%20DETERMINANTS.pdf)
- The movement demonstrations provide examples of basic steps, lunges and recovery. [MIT OpenCourseWare PE.740 Fencing, Spring 2007](https://ocw.mit.edu/courses/pe-740-fencing-spring-2007/pages/video/)

The numerical grading ranges are defined in [the project rubric](fencing_coach/rubric.json).
