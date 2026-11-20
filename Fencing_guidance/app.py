"""Run: python -m streamlit run app.py"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from fencing_coach.analysis import analyze_frames
from fencing_coach.demo import synthetic_sequence
from fencing_coach.pipeline import analyze_camera, analyze_video, save_outputs
from fencing_coach.schema import ACTION_NAMES, AnalysisConfig
from fencing_coach.vision import DEFAULT_POSE_MODEL, DEFAULT_SWORD_MODEL

st.set_page_config(page_title="Fencing Practice Review", page_icon="🤺", layout="wide")
st.title("Fencing Practice Review")
st.write("Upload or record a solo drill to get a PDF with measurements and points to review.")

with st.sidebar:
    st.header("Capture setup")
    weapon = st.selectbox("Weapon", ["foil", "epee", "sabre"])
    hand = st.selectbox("Weapon hand", ["right", "left"])
    front = st.selectbox("Front leg", ["right", "left"])
    static = st.checkbox("The camera stays fixed throughout", value=True)
    level = st.checkbox("The camera is level", value=True)
    mirrored = st.checkbox(
        "The source video is mirrored",
        value=False,
        help="Enable only if the actual saved input is mirrored. The system unmirrors before estimating anatomical sides.",
    )
    use_sword = st.checkbox("Show sword outline", value=False)
    if use_sword:
        st.caption(
            "Install the packages in requirements-sword.txt to show the sword outline in the video overlay."
        )
    fps = st.select_slider("Pose sampling limit (FPS)", [10, 15, 20, 25, 30], value=30)

st.info(
    "Hold en garde for at least 1.2 seconds before moving. Keep your whole body in frame, including both feet and your weapon arm. A fixed camera at the side or at an angle is best for footwork."
)
mode = st.radio("Input", ["Upload a video", "Local webcam", "Sample drill"], horizontal=True)
uploaded = None
camera_id, seconds = 0, 15
if mode == "Upload a video":
    uploaded = st.file_uploader(
        "Solo drill video (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"]
    )
    st.caption("The app processes up to five minutes on the computer running Python.")
elif mode == "Local webcam":
    st.write(
        "The webcam must be connected to the computer running Python. On a remote server, record on your device and use video upload."
    )
    camera_id = st.number_input("Local camera device ID", min_value=0, max_value=8, value=0)
    seconds = st.slider("Capture duration (seconds)", 5, 60, 15)
else:
    st.caption("This sample uses simulated joint positions.")

if st.button(
    "Generate report",
    type="primary",
    disabled=mode == "Upload a video" and uploaded is None,
):
    st.session_state.pop("report", None)
    config = AnalysisConfig(
        weapon_side=hand,
        front_side=front,
        weapon=weapon,
        camera_static=static,
        camera_level=level,
        mirrored_input=mirrored,
        sample_fps=float(fps),
    )
    try:
        with tempfile.TemporaryDirectory(prefix="fencing_session_") as directory:
            root = Path(directory)
            output = root / "report"
            bar = st.progress(0.0, text="Preparing local analysis")
            if mode == "Sample drill":
                frames = synthetic_sequence(side=front)
                result, features = analyze_frames(
                    frames, config, "Sample lunge and recovery", simulated=True
                )
                artifacts = save_outputs(result, features, frames, output)
            else:
                if not DEFAULT_POSE_MODEL.is_file():
                    raise FileNotFoundError(
                        "Pose model missing. In a terminal, run: python tools/download_pose_model.py"
                    )
                sword_model = DEFAULT_SWORD_MODEL if use_sword else None
                if mode == "Upload a video":
                    source = root / ("input" + Path(uploaded.name).suffix.lower())
                    source.write_bytes(uploaded.getvalue())
                    result, artifacts = analyze_video(
                        source,
                        output,
                        config,
                        sword_model=sword_model,
                        progress=lambda value, message: bar.progress(value, text=message),
                        source_name=Path(uploaded.name).name,
                    )
                else:
                    live = st.empty()

                    def preview(image, elapsed):
                        live.image(
                            image, channels="BGR", caption=f"Recording: {elapsed:.1f} seconds"
                        )

                    result, artifacts = analyze_camera(
                        output,
                        config,
                        int(camera_id),
                        seconds,
                        sword_model=sword_model,
                        preview=preview,
                    )
            bar.progress(1.0, text="Report ready")
            payload = {name: path.read_bytes() for name, path in artifacts.items()}
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, path in artifacts.items():
                    archive.writestr(path.name, payload[name])
            st.session_state["report"] = {
                "result": result.to_dict(),
                "files": payload,
                "zip": bundle.getvalue(),
            }
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        st.error(f"The report could not be generated: {exc}")

if "report" in st.session_state:
    saved = st.session_state["report"]
    result = saved["result"]
    st.subheader("Observed movements")
    st.write(
        f"Calibration: {'accepted' if result['calibration']['valid'] else 'rejected'} | Estimated view: {result['calibration']['view']} | Usable sampled frames: {result['observed_fraction']:.0%}"
    )
    rows = [
        {
            "Movement": ACTION_NAMES[item["action"]["kind"]],
            "Start (s)": round(item["action"]["start"], 2),
            "End (s)": round(item["action"]["end"], 2),
            "Overall /100": f"{item['total']:.1f}" if item["total"] is not None else "Withheld",
            "Weighted evidence": f"{item['coverage']:.0%}",
        }
        for item in result["actions"]
    ]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.warning(
            "No supported movement was identified. The PDF explains the capture limitations."
        )
    left, right = st.columns(2)
    left.download_button(
        "Download coaching PDF",
        saved["files"]["pdf"],
        "coaching_report.pdf",
        "application/pdf",
        type="primary",
    )
    right.download_button(
        "Download all results (ZIP)",
        saved["zip"],
        "fencing_evidence.zip",
        "application/zip",
    )
    with st.expander("Capture and scoring notes"):
        for message in result["warnings"]:
            st.write(message)
