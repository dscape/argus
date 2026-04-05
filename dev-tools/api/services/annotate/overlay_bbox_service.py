"""Service for overlay bbox annotation ground truth."""

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from pipeline.overlay.scanner import (
    _refine_alignment,
    check_alternating_pattern,
    compute_grid_regularity,
)
from pipeline.paths import GROUND_TRUTH_PATH, VIDEOS_DIR
from pipeline.paths import frame_path as _frame_path

logger = logging.getLogger(__name__)

# Cache discovered frame paths to avoid rescanning 73K+ directories.
# Invalidated when ground truth changes or after TTL expires.
_frame_cache: list[tuple[str, str]] | None = None  # [(video_id, label), ...]
_frame_cache_time: float = 0
_CACHE_TTL = 300  # 5 minutes

# Legacy path used as fallback during migration
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_LEGACY_FRAMES_DIR = _PROJECT_ROOT / "data" / "overlay" / "dataset" / "frames"
_LEGACY_GT_PATH = _LEGACY_FRAMES_DIR / "ground_truth.json"


def _load_ground_truth() -> dict:
    if GROUND_TRUTH_PATH.exists():
        return json.loads(GROUND_TRUTH_PATH.read_text())
    # Fall back to legacy location
    if _LEGACY_GT_PATH.exists():
        return json.loads(_LEGACY_GT_PATH.read_text())
    return {}


def _save_ground_truth(data: dict) -> None:
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = GROUND_TRUTH_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(GROUND_TRUTH_PATH)


def _discover_frame_paths() -> list[tuple[str, str]]:
    """Scan filesystem for hires frames. Result is cached in-memory."""
    global _frame_cache, _frame_cache_time  # noqa: PLW0603
    now = time.monotonic()
    if _frame_cache is not None and (now - _frame_cache_time) < _CACHE_TTL:
        return _frame_cache

    found: list[tuple[str, str]] = []

    if VIDEOS_DIR.exists():
        base = str(VIDEOS_DIR)
        try:
            for entry in os.scandir(base):
                if not entry.is_dir(follow_symlinks=False):
                    continue
                hires = os.path.join(entry.path, "hires")
                if not os.path.isdir(hires):
                    continue
                vid = entry.name
                try:
                    for img in os.scandir(hires):
                        if img.name.endswith(".jpg"):
                            found.append((vid, img.name[:-4]))
                except OSError:
                    continue
        except OSError:
            pass

    # Fall back to legacy dir
    if not found and _LEGACY_FRAMES_DIR.exists():
        for video_dir in _LEGACY_FRAMES_DIR.iterdir():
            if not video_dir.is_dir():
                continue
            vid = video_dir.name
            for img_path in video_dir.glob("*.jpg"):
                found.append((vid, img_path.stem))

    _frame_cache = found
    _frame_cache_time = now
    return found


def _invalidate_frame_cache() -> None:
    global _frame_cache  # noqa: PLW0603
    _frame_cache = None


def list_frames() -> list[dict]:
    """List all hires frames with annotation status.

    Returns unannotated frames in random order (to avoid sampling bias),
    followed by annotated frames sorted by annotation time.
    """
    gt = _load_ground_truth()
    paths = _discover_frame_paths()

    frames = []
    for video_id, label in paths:
        key = f"{video_id}/{label}"
        annotation = gt.get(key)
        frames.append(
            {
                "key": key,
                "video_id": video_id,
                "label": label,
                "annotated": annotation is not None,
                "has_overlay": (
                    annotation.get("has_overlay")
                    if annotation else None
                ),
                "bbox": (
                    annotation.get("bbox")
                    if annotation else None
                ),
            }
        )

    # Split into annotated vs unannotated
    annotated = [f for f in frames if f["annotated"]]
    unannotated = [f for f in frames if not f["annotated"]]

    # Randomize unannotated to avoid sampling bias
    random.shuffle(unannotated)

    # Annotated at end, sorted by annotation time (most recent first)
    annotated.sort(
        key=lambda f: gt.get(f["key"], {}).get("annotated_at", ""),
        reverse=True,
    )

    return unannotated + annotated


def get_frame_path(video_id: str, label: str) -> Path | None:
    """Resolve and validate a frame file path (hires preferred)."""
    # New layout
    path = _frame_path(video_id, "hires", label)
    if path.exists():
        return path
    # Legacy
    legacy = _LEGACY_FRAMES_DIR / video_id / f"{label}.jpg"
    if legacy.exists():
        return legacy
    return None


def refine_bbox(
    frame_path: Path, rough_bbox: list[int]
) -> dict:
    """Lightly refine a rough bbox: enforce square and nudge a few pixels.

    The user's drawing is trusted — we only make minimal adjustments:
    - Enforce square aspect ratio (use max of w/h, re-center)
    - Nudge up to 5px to maximize grid alignment
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return {"error": "Could not read frame"}
    fh, fw = frame.shape[:2]

    x, y, w, h = rough_bbox

    # Enforce square: use max dimension, re-center
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x = max(0, cx - side // 2)
    y = max(0, cy - side // 2)
    # Clamp to frame
    if x + side > fw:
        x = fw - side
    if y + side > fh:
        y = fh - side
    x = max(0, x)
    y = max(0, y)
    side = min(side, fw - x, fh - y)

    squared_bbox = (x, y, side, side)

    # Light nudge only — max 5px shift to improve grid alignment
    refined = _refine_alignment(frame, squared_bbox, max_shift=5)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bx, by, bw, bh = refined
    region = gray[by : by + bh, bx : bx + bw]
    best_score = compute_grid_regularity(region)
    has_pattern = check_alternating_pattern(region)
    best_bbox = refined

    return {
        "bbox": list(best_bbox),
        "score": round(best_score, 4),
        "has_pattern": has_pattern,
        "original": rough_bbox,
    }


def save_annotation(
    frame_key: str,
    has_overlay: bool,
    bbox: list[int] | None,
    notes: str = "",
) -> dict:
    """Save or update an annotation."""
    gt = _load_ground_truth()

    video_id, label = frame_key.split("/", 1)
    path = get_frame_path(video_id, label)
    if path is None:
        return {"error": f"Frame not found: {frame_key}"}

    img = cv2.imread(str(path))
    fh, fw = img.shape[:2] if img is not None else (0, 0)

    entry: dict = {
        "image": f"{video_id}/{label}.jpg",
        "has_overlay": has_overlay,
        "bbox": bbox,
        "frame_width": fw,
        "frame_height": fh,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
    if notes:
        entry["notes"] = notes

    gt[frame_key] = entry
    _save_ground_truth(gt)
    return {"saved": True, "key": frame_key}


def delete_annotation(frame_key: str) -> bool:
    """Remove an annotation."""
    gt = _load_ground_truth()
    if frame_key not in gt:
        return False
    del gt[frame_key]
    _save_ground_truth(gt)
    return True
