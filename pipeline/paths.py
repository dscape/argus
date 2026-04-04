"""Central path resolution for all video assets.

Single source of truth for video files, frame images, and ground-truth
annotations.  All code that constructs asset paths should import from here
instead of computing paths locally.

Layout::

    data/videos/{video_id}/
        {video_id}.mp4
        lores/25pct.jpg          480x360  (YouTube hq thumbnails)
        lores/50pct.jpg
        lores/75pct.jpg
        hires/25pct.jpg          1280x720 (YouTube maxres thumbnails)
        hires/50pct.jpg
        hires/75pct.jpg
        fullres/25pct.jpg        1920x1080 (yt-dlp / ffmpeg)
        fullres/50pct.jpg
        fullres/75pct.jpg
"""

import glob as glob_mod
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"

Tier = Literal["lores", "hires", "fullres"]
FRAME_LABELS = ("25pct", "50pct", "75pct")

# Ground-truth overlay bbox annotations (migrated from overlay/dataset/frames/)
GROUND_TRUTH_PATH = VIDEOS_DIR / "ground_truth.json"

# ── Legacy paths (used by find_* fallbacks during migration) ──────────

_LEGACY_SCREENING_FRAMES = PROJECT_ROOT / "data" / "screening" / "dataset" / "frames"
_LEGACY_OVERLAY_FRAMES = PROJECT_ROOT / "data" / "overlay" / "dataset" / "frames"


# ── Canonical paths ───────────────────────────────────────────────────


def video_dir(video_id: str) -> Path:
    """Root directory for all assets of one video."""
    return VIDEOS_DIR / video_id


def video_file(video_id: str) -> Path:
    """Canonical path for the downloaded video file."""
    return VIDEOS_DIR / video_id / f"{video_id}.mp4"


def frame_dir(video_id: str, tier: Tier) -> Path:
    """Directory for a resolution tier's frames."""
    return VIDEOS_DIR / video_id / tier


def frame_path(video_id: str, tier: Tier, label: str) -> Path:
    """Path to a single frame image."""
    return VIDEOS_DIR / video_id / tier / f"{label}.jpg"


# ── Finders (with legacy fallback) ────────────────────────────────────


def find_video_file(video_id: str) -> Path | None:
    """Locate a downloaded video, checking new layout then legacy."""
    # New layout
    new = video_file(video_id)
    if new.exists():
        return new

    # Legacy: data/videos/{channel_handle}/{video_id}.mp4
    for ext in ("mp4", "mkv", "webm"):
        pattern = str(VIDEOS_DIR / "*" / f"{video_id}.{ext}")
        matches = glob_mod.glob(pattern)
        if matches:
            return Path(matches[0])

    return None


def find_frame(video_id: str, tier: Tier, label: str) -> Path | None:
    """Locate a frame, checking new layout then legacy directories."""
    # New layout
    new = frame_path(video_id, tier, label)
    if new.exists():
        return new

    # Legacy fallbacks
    if tier == "lores":
        legacy = _LEGACY_SCREENING_FRAMES / video_id / f"{label}.jpg"
        if legacy.exists():
            return legacy
    elif tier in ("hires", "fullres"):
        legacy = _LEGACY_OVERLAY_FRAMES / video_id / f"{label}.jpg"
        if legacy.exists():
            return legacy

    return None


def asset_status(video_id: str) -> dict:
    """Report which assets exist for a video.

    Returns::

        {
            "video": bool,
            "lores": ["25pct", ...],
            "hires": ["25pct", ...],
            "fullres": ["25pct", ...],
        }
    """
    result: dict = {
        "video": find_video_file(video_id) is not None,
        "lores": [],
        "hires": [],
        "fullres": [],
    }
    for tier in ("lores", "hires", "fullres"):
        for label in FRAME_LABELS:
            if find_frame(video_id, tier, label) is not None:  # type: ignore[arg-type]
                result[tier].append(label)
    return result
