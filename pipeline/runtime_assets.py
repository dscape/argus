"""Validation for committed runtime model weights.

Used by host make targets to fail fast when Git LFS blobs were not fetched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.paths import PROJECT_ROOT

_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


@dataclass(frozen=True)
class RuntimeAsset:
    name: str
    path: Path


SCREENING_WEIGHTS = RuntimeAsset(
    name="screening classifier weights",
    path=PROJECT_ROOT / "weights" / "screening" / "best.pt",
)
OVERLAY_PIECE_CLASSIFIER_WEIGHTS = RuntimeAsset(
    name="overlay piece-classifier weights",
    path=PROJECT_ROOT / "weights" / "overlay" / "best.pt",
)
OVERLAY_YOLO_WEIGHTS = RuntimeAsset(
    name="overlay YOLO detector weights",
    path=PROJECT_ROOT / "weights" / "overlay_yolo" / "best.pt",
)
RUNTIME_WEIGHT_ASSETS = (
    SCREENING_WEIGHTS,
    OVERLAY_PIECE_CLASSIFIER_WEIGHTS,
    OVERLAY_YOLO_WEIGHTS,
)


def is_git_lfs_pointer(path: Path) -> bool:
    """Return True when a file is a Git LFS pointer instead of the real blob."""
    if not path.exists() or not path.is_file():
        return False

    with path.open("rb") as handle:
        return handle.read(len(_GIT_LFS_POINTER_PREFIX)) == _GIT_LFS_POINTER_PREFIX


def ensure_runtime_asset(asset: RuntimeAsset) -> Path:
    """Validate one runtime asset and return its absolute path."""
    path = asset.path.expanduser().resolve()
    display_path = _display_path(path)
    include_arg = _git_lfs_include_arg(path)

    if not path.exists():
        hint = (
            ""
            if include_arg is None
            else f" Fetch committed model weights with Git LFS: `git lfs install && "
            f"git lfs pull --include='{include_arg}'`."
        )
        raise FileNotFoundError(f"Missing {asset.name} at {display_path}.{hint}")

    if is_git_lfs_pointer(path):
        hint = (
            ""
            if include_arg is None
            else f" Install Git LFS and fetch the blobs: `git lfs install && git lfs "
            f"pull --include='{include_arg}'`."
        )
        raise RuntimeError(
            f"{display_path} is still a Git LFS pointer, not the real model weights.{hint}"
        )

    return path


def ensure_default_runtime_assets() -> tuple[Path, ...]:
    """Validate all committed runtime weights required by the default stack."""
    return tuple(ensure_runtime_asset(asset) for asset in RUNTIME_WEIGHT_ASSETS)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _git_lfs_include_arg(path: Path) -> str | None:
    try:
        rel_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return f"{rel_path.parent}/*"


def main() -> int:
    try:
        validated = ensure_default_runtime_assets()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Runtime asset check failed: {exc}")
        return 1

    print("Verified committed runtime weights:")
    for path in validated:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
