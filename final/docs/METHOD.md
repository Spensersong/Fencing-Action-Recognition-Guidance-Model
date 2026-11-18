# How the measurements work

MediaPipe provides 33 body landmarks for each sampled frame. The program uses them to measure positions and angles, recognise a movement over time, and compare that movement with the ranges in [the rubric](../fencing_coach/rubric.json). The optional sword model produces a separate outline for the overlay.

## Landmarks and starting position

A frame contains two sets of 33 rows. Each row has x, y, z, visibility and presence values.

| Coordinates | Meaning and use |
| --- | --- |
| Image | x and y are fractions of width and height. The program converts them to pixels for measuring image displacement and drawing the overlay. |
| World | The pose model estimates 3D positions in metres around the midpoint of the hips. The program uses these positions for joint angles and relative body geometry. |

The knee angle uses the hip, knee and ankle. The elbow angle uses the shoulder, elbow and wrist of the selected weapon arm. At joint B, divide the dot product of vectors A − B and C − B by the product of their lengths, then take the arccosine.

During the opening 1.2 seconds, the program looks for a stable guard with visible shoulders, hips, knees and ankles. The default visibility and presence threshold is 0.65. The guard needs at least five usable samples spanning 0.45 seconds and at least 80% usable coverage. The initial guard sets the forward direction from the rear ankle to the front ankle and provides two scales.

| Report unit | Scale |
| --- | --- |
| `leg` | Average estimated 3D leg length from hip through knee to ankle. |
| `torso` | Projected shoulder-midpoint to hip-midpoint distance in the initial image. |

World coordinates are relative to the hips and therefore cannot show travel along the piste on their own. The program measures travel in the image if the camera is fixed and the forward direction has sufficient horizontal projection. A projection ratio of at least 0.82 is classified as side view. A ratio from 0.45 up to 0.82 is classified as oblique. Lower ratios are classified as frontal. These labels are estimates based on the pose model.

## Recognising a drill

| Movement | What the program looks for |
| --- | --- |
| En garde | A sustained hold near the starting guard. |
| Advance | Both feet progress forward along the calibrated direction. |
| Retreat | Both feet progress backward. |
| Straight arm extension | The weapon-side elbow opens and reach increases while foot spacing stays stable. |
| Lunge | Foot spacing increases and the front and rear legs take on different configurations over time. |
| Recovery to en garde | The body returns from a lunge to a stable guard. |
| Advance-lunge | A detected advance followed closely by a lunge. Both movements contribute to the score. |

The recognition rules run before scoring, so an action can be recognised even if its measurements fall outside the scoring band. An interval that does not fit the supported patterns is marked unclassified.

A missing pose, multiple people, crop, an abrupt shift or an implausible change in limb length can remove a frame from analysis. A gap of more than 0.20 seconds separates observations. The program smooths features within each usable sequence. Two onsets have to be separated by more than 0.07 seconds or two sample intervals, whichever is greater, before their order can be scored. Camera movement, poor framing and occlusion reduce the measurements the report can use.

## Interpreting a measurement

| Measurement | How to read it |
| --- | --- |
| Joint angle | Estimated 3D angle near the peak or finish shown in the report. |
| Trunk inclination | Angle from camera vertical. Use a level camera to interpret it. |
| Stance ratio | Ankle spacing along the calibrated forward direction, in leg lengths. |
| Stance error | Absolute difference between the final stance ratio and the starting guard. |
| Knee offset | Distance between the front knee and ankle along the forward direction, in leg lengths. |
| Arm lead | Time of front-foot onset minus time of arm-extension onset. |
| Foot order | Time of following-foot onset minus time of leading-foot onset. |
| Hip excursion | Vertical image movement of the hip midpoint, in projected torso lengths. |
| Trunk variation | Standard deviation of the measured trunk angle over the interval. |
| Reach-path ratio | Total change along the measured reach path divided by its range. Reversals increase this ratio. |

Peak and finish measurements can use several nearby frames. An interval measurement, such as trunk variation, describes the movement as a whole. Its timestamp is a marker for reviewing the recording.

## Calculating points

Each measurement has a reference band, a pair of outer bounds and a weight. A value within the band earns all the points for that item. The points decrease linearly toward zero at the outer bounds. The [rubric file](../fencing_coach/rubric.json) gives the settings and associated review cues for each drill.

A category is scored when every item in that category is available. The total also requires an observed finish. Weighted coverage shows how many of the 100 possible points had a usable measurement, regardless of the points earned. For an advance-lunge, the advance and lunge each contribute half of every category.

Report suggestions are tied to measured deviations from these reference ranges. Review the recording at the indicated times and compare scores only for the same drill, rubric and capture setup. Foil and epee use the classical extension settings. Sabre arm-extension measurements are left unscored where those settings do not apply.

Movement demonstrations, descriptions of the classical lunge and model documentation are listed in [Models and references](../THIRD_PARTY_NOTICES.md). The numerical bands and weights are project settings.
