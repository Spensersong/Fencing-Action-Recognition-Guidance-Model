# Synthetic software examples

All files in this directory use analytically specified landmarks, not videos of a real athlete and not labelled training data. Every generated report is marked **SIMULATED DATA** on every page.

`Fencing_Coaching_Report_Example.pdf` is the side-view report. Its fixture contains a specified trunk inclination outside the chosen prototype band, so the report demonstrates an evidence-linked review cue as well as the score breakdown.

`side_view/` contains the PDF, structured results, measurements and synthetic landmarks. `frontal_view/` rotates the same analytical kinematics toward the camera. Supported 3D posture observations remain available, while translation-dependent measurements and the complete /100 grade are withheld. This illustrates software observability rules, not proof that MediaPipe recovers real frontal poses accurately.

Reproduce both examples from the repository root:

```bash
python tools/generate_examples.py
```

The held-out evaluator rejects results marked as simulated. Do not remove that flag or use these samples to claim real-world recognition accuracy or professional coach agreement.
