"""Ensure the chess-positions dataset is available for overlay tooling."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OVERLAY_DIR = _PROJECT_ROOT / "data" / "overlay"
_TRAIN_DIR = _OVERLAY_DIR / "train"
_VAL_DIR = _OVERLAY_DIR / "val"
_DATASET_ID = "koryakinp/chess-positions"
_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class DatasetStatus:
    train_ready: bool
    val_ready: bool

    @property
    def ready(self) -> bool:
        return self.train_ready and self.val_ready

    @property
    def missing_splits(self) -> list[str]:
        missing: list[str] = []
        if not self.train_ready:
            missing.append("data/overlay/train")
        if not self.val_ready:
            missing.append("data/overlay/val")
        return missing


def _has_images(directory: Path) -> bool:
    if not directory.exists():
        return False

    return any(
        path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        for path in directory.rglob("*")
    )


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def get_dataset_status(overlay_dir: Path = _OVERLAY_DIR) -> DatasetStatus:
    """Return whether the expected train and val splits are populated."""
    return DatasetStatus(
        train_ready=_has_images(overlay_dir / "train"),
        val_ready=_has_images(overlay_dir / "val"),
    )


def _find_split_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    priority = {name.casefold(): index for index, name in enumerate(names)}
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.casefold() in priority
    ]
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda path: (priority[path.name.casefold()], len(path.parts), str(path)),
    )[0]


def install_downloaded_dataset(
    download_dir: Path,
    overlay_dir: Path = _OVERLAY_DIR,
) -> DatasetStatus:
    """Copy downloaded chess-positions splits into data/overlay/{train,val}."""
    train_source = _find_split_dir(download_dir, ("train",))
    val_source = _find_split_dir(download_dir, ("val", "test"))

    if train_source is None:
        raise FileNotFoundError("Downloaded chess-positions archive is missing a train/ split")
    if val_source is None:
        raise FileNotFoundError(
            "Downloaded chess-positions archive is missing a val/ or test/ split"
        )

    train_target = overlay_dir / "train"
    val_target = overlay_dir / "val"
    train_target.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(train_source, train_target, dirs_exist_ok=True)
    shutil.copytree(val_source, val_target, dirs_exist_ok=True)

    return get_dataset_status(overlay_dir)


def _print_manual_setup() -> None:
    print("Manual setup:")
    print(
        "  1. Download the Kaggle dataset 'koryakinp/chess-positions' "
        "(https://www.kaggle.com/datasets/koryakinp/chess-positions)."
    )
    print("  2. Put its train/ split in data/overlay/train/.")
    print("  3. Put its test/ split in data/overlay/val/.")


def _download_dataset() -> bool:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        print("Kaggle CLI not found; cannot auto-download chess-positions.")
        print("Install it first, e.g. `python3 -m pip install kaggle`, and configure credentials.")
        return False

    with tempfile.TemporaryDirectory(prefix="argus_chess_positions_") as tmp:
        temp_dir = Path(tmp)
        print(f"Downloading {_DATASET_ID} via Kaggle CLI...")
        result = subprocess.run(
            [
                kaggle,
                "datasets",
                "download",
                "-d",
                _DATASET_ID,
                "-p",
                str(temp_dir),
                "--unzip",
                "--force",
            ],
            cwd=_PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            print("Kaggle download failed.")
            return False

        status = install_downloaded_dataset(temp_dir)
        if not status.ready:
            print("Downloaded dataset is still incomplete after extraction.")
            return False

    print(
        "Installed chess-positions into "
        f"data/overlay/train ({_count_images(_TRAIN_DIR)} images) and "
        f"data/overlay/val ({_count_images(_VAL_DIR)} images)."
    )
    return True


def ensure_chess_positions(prompt: bool = False, assume_yes: bool = False) -> bool:
    """Ensure chess-positions train/val data exists, optionally prompting to download."""
    status = get_dataset_status()
    if status.ready:
        return True

    missing = ", ".join(status.missing_splits)
    print(f"Missing chess-positions data: {missing}")

    if assume_yes:
        downloaded = _download_dataset()
        if not downloaded:
            _print_manual_setup()
        return downloaded

    if not prompt:
        _print_manual_setup()
        return False

    if not sys.stdin.isatty():
        print("Non-interactive shell; skipping chess-positions download prompt.")
        _print_manual_setup()
        return False

    answer = input(
        "Download chess-positions now via Kaggle CLI so overlay evaluation works? [y/N] "
    ).strip().lower()
    if answer not in {"y", "yes"}:
        _print_manual_setup()
        return False

    downloaded = _download_dataset()
    if downloaded:
        return True

    _print_manual_setup()
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure the chess-positions dataset is available under data/overlay/.",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Ask before downloading when train/ or val/ is missing.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Download immediately without prompting.",
    )
    args = parser.parse_args()

    ensure_chess_positions(prompt=args.prompt, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
