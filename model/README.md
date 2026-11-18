# Fencing Action Recognition and Guidance

I worked on this project to help beginners review their fencing practice from video. If you film a lunge, for example, you can check the timing of your arm and front foot and go back to that part of the clip. The program uses MediaPipe to estimate joint positions and OpenCV to read the video. It checks those positions over time and puts the measurements, scores and practice suggestions in a PDF.

## Run the app

On Windows, install 64-bit Python 3.11 and double-click `start_windows.bat` in the project folder. The first run installs the required packages and opens the app in your browser.

On macOS or Linux, open a terminal in the project folder and run the following commands with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

The app runs on the computer where you start Python. Use **Upload a video** to analyse a saved clip, **Local webcam** to record with a camera connected to that computer, or **Sample drill** to try the report with simulated joint positions.

## Record a drill

1. Place the camera at the side or slightly to the front. Keep it fixed and level. Frame one fencer and keep both feet, knees and the weapon arm visible.
2. Start with at least 1.2 seconds of still en garde so the program can set the starting stance. Record the whole drill, including the finish. Use normal playback speed and aim for 30 FPS.
3. Select the weapon, weapon hand and front leg. Turn on the mirror setting if the saved video is mirrored.
4. Click **Generate report**, then **Download coaching PDF**. For the other files, choose **Download all results (ZIP)**.

The program looks for en garde, advance, retreat, straight arm extension, lunge, recovery to en garde and advance-lunge. A side or angled view shows forward and backward steps more clearly. A front view can still give some joint angles. The report leaves a measurement unassessed when the view, timing or visibility is insufficient.

## Read the report

Each movement has a start and end time, a table of measurements and points to review. The lunge section can show the estimated front-knee angle at the peak and the time difference between the arm and front-foot onsets. Use the timestamps to find the corresponding part of the clip.

| Category | Points |
| --- | ---: |
| Joint configuration | 45 |
| Timing and coordination | 25 |
| Movement control | 20 |
| Fluency and consistency | 10 |
| **Total** | **100** |

The total requires a complete movement and a measurement for every scored item. The score describes how closely the movement matches the reference ranges for that drill. The ranges are project settings for solo practice and have yet to be validated against independent coaching assessments.

[Side-view report](examples/side_view.pdf) · [Front-view report](examples/frontal_view.pdf) · [How the measurements work](docs/METHOD.md)

## Optional sword outline

I used Label Studio to annotate images for the sword model. The trained weights in `best(1).pt` identify the sword as a single region in the frame. The app uses that region in the video overlay. The movement score uses the body landmarks.

To enable the overlay, install the extra dependency in the same Python environment and restart the app.

```bash
python -m pip install -r requirements-sword.txt
```

On Windows, substitute `.venv\Scripts\python.exe` for `python` if you started the app using the batch file. Select **Show sword outline** in the sidebar.

## Command line

```bash
python -m fencing_coach.cli --video my_drill.mp4 --output outputs/my_drill
```

[Command-line options and output files](docs/COMMAND_LINE.md) · [Testing and evaluation](docs/VALIDATION.md) · [Models and references](THIRD_PARTY_NOTICES.md)
