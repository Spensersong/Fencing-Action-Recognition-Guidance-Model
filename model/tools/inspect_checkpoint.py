"""Inspect checkpoint strings without executing pickle data or importing torch."""

import argparse
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if len(names) != 1:
            raise ValueError("Expected one PyTorch checkpoint data.pkl member.")
        strings = [
            str(arg)
            for op, arg, _ in pickletools.genops(archive.read(names[0]))
            if op.name in {"GLOBAL", "BINUNICODE", "SHORT_BINUNICODE"}
        ]
    result = {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "inspection": "Static pickle string inspection only; no checkpoint was executed.",
        "segmentation_model_symbol_present": any("SegmentationModel" in s for s in strings),
    }
    for name in ("version", "date", "names", "task"):
        if name in strings:
            index = strings.index(name)
            result[name + "_following_string"] = (
                strings[index + 1] if index + 1 < len(strings) else None
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "best(1).pt",
    )
    args = parser.parse_args()
    print(json.dumps(inspect(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
