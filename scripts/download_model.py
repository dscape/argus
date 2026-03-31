"""Download DINOv2-base weights to weights/dinov2-base/ for offline use."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ID = "facebook/dinov2-base"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "weights" / "dinov2-base"


def main() -> None:
    if (LOCAL_DIR / "config.json").exists() and any(LOCAL_DIR.glob("*.safetensors")):
        print(f"DINOv2-base already present at {LOCAL_DIR}")
        return

    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO_ID} → {LOCAL_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        ignore_patterns=["*.h5", "*.ot", "*.msgpack", "tf_*", "flax_*"],
    )
    print("Done.")


if __name__ == "__main__":
    sys.exit(main() or 0)
