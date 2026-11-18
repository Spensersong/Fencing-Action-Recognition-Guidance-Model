"""Fetch the official MediaPipe Full Pose Landmarker asset, with a checksum."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_full.task",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        print(f"Already present: {args.output}")
        print("SHA-256:", hashlib.sha256(args.output.read_bytes()).hexdigest())
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(URL, timeout=90) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    if stream.tell() > 30_000_000:
                        raise ValueError("The model response exceeded the expected asset size.")
        data = temporary.read_bytes()
        if len(data) < 1_000_000 or not zipfile.is_zipfile(temporary):
            raise ValueError("The response is not a MediaPipe task model bundle.")
        manifest = args.output.parent / "manifest.json"
        if manifest.is_file():
            import json

            expected = json.loads(manifest.read_text())["pose_landmarker_full.task"]["sha256"]
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(
                    "The downloaded model did not match the bundled manifest checksum."
                )
        os.replace(temporary, args.output)
        print(f"Downloaded: {args.output}")
        print("SHA-256:", hashlib.sha256(data).hexdigest())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
