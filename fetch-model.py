"""Fetch and verify the two GGUF files from the pinned public QnVoice release."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
RELEASE = (
    "https://huggingface.co/CholoAsis/qnvoice-cebuano-translation"
    "/resolve/27efd4da6be61a51219234ca584586cb084a6133"
)


def fetch(destination, info):
    if destination.is_file() and destination.stat().st_size == info["bytes"]:
        with destination.open("rb") as current:
            if hashlib.file_digest(current, "sha256").hexdigest() == info["sha256"]:
                print(f"Verified existing {destination}")
                return

    print(f"Downloading {destination.name} ({info['bytes']:,} bytes)", flush=True)
    digest = hashlib.sha256()
    count = 0
    partial = None
    try:
        with NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as output:
            partial = Path(output.name)
            with urlopen(f"{RELEASE}/{destination.name}", timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > info["bytes"]:
                        raise ValueError(f"Unexpected file size for {destination.name}")
                    digest.update(chunk)
                    output.write(chunk)
        if count != info["bytes"] or digest.hexdigest() != info["sha256"]:
            raise ValueError(f"Size or SHA-256 mismatch for {destination.name}")
        partial.replace(destination)
        print(f"Verified {destination}", flush=True)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "benchmarks/results.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, info in manifest["files"].items():
        fetch(args.output_dir / filename, info)


if __name__ == "__main__":
    main()
