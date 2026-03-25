"""Fetch YouTube auto-generated frame thumbnails.

YouTube serves 4 auto-generated thumbnails per video at known URLs, with no
API quota cost. These are used for screening inspection and AI classification.

Images are cached on disk under data/frame_cache/{video_id}/ so that
re-extraction and re-training never need to re-download from YouTube.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,15}$")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAME_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "frame_cache")

# YouTube auto-generated frame URLs (publicly accessible, no API quota).
# hq variants serve 480x360 (vs 120x90 for plain 1/2/3.jpg).
# 0.jpg = default thumbnail (480x360), hq1/hq2/hq3 ~ 25%/50%/75% (480x360).
FRAME_URLS = [
    ("https://img.youtube.com/vi/{video_id}/0.jpg", "thumb"),
    ("https://img.youtube.com/vi/{video_id}/hq1.jpg", "25pct"),
    ("https://img.youtube.com/vi/{video_id}/hq2.jpg", "50pct"),
    ("https://img.youtube.com/vi/{video_id}/hq3.jpg", "75pct"),
]

# Mapping for backward compatibility with old label format
_LABEL_DISPLAY = {"thumb": "thumb", "25pct": "25%", "50pct": "50%", "75pct": "75%"}

_MIN_FRAME_WIDTH = 100


def _cache_path(video_id: str, label: str) -> str:
    """Return the on-disk cache path for a single frame."""
    return os.path.join(FRAME_CACHE_DIR, video_id, f"{label}.jpg")


def _load_cached_frame(video_id: str, label: str) -> np.ndarray | None:
    """Try to load a frame from the disk cache."""
    path = _cache_path(video_id, label)
    if not os.path.exists(path):
        return None
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is not None and frame.shape[1] >= _MIN_FRAME_WIDTH:
        return frame
    return None


def _save_frame_to_cache(video_id: str, label: str, frame: np.ndarray) -> None:
    """Save a frame to the disk cache."""
    path = _cache_path(video_id, label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, frame)


def fetch_single_frame(url: str) -> np.ndarray | None:
    """Fetch a single image URL and return as a BGR numpy array."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None and frame.shape[1] >= _MIN_FRAME_WIDTH:
                return frame
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        pass
    return None


def fetch_youtube_frames(video_id: str) -> list[tuple[np.ndarray, str]]:
    """Fetch all 4 YouTube auto-generated frames, using disk cache when available.

    Returns list of (frame_bgr, display_label) tuples sorted in canonical order.
    Frames are cached under data/frame_cache/{video_id}/ on first fetch.
    """
    if not _VIDEO_ID_RE.match(video_id):
        logger.warning(f"Invalid video_id format: {video_id!r}")
        return []

    results: list[tuple[np.ndarray, str]] = []
    to_fetch: list[tuple[str, str]] = []  # (url, label) pairs that need downloading

    # First pass: check disk cache
    for url_template, label in FRAME_URLS:
        cached = _load_cached_frame(video_id, label)
        if cached is not None:
            results.append((cached, _LABEL_DISPLAY[label]))
        else:
            to_fetch.append((url_template.format(video_id=video_id), label))

    # Second pass: download missing frames in parallel
    if to_fetch:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(fetch_single_frame, url): label
                for url, label in to_fetch
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    frame = future.result()
                    if frame is not None:
                        _save_frame_to_cache(video_id, label, frame)
                        results.append((frame, _LABEL_DISPLAY[label]))
                except Exception as e:
                    logger.warning(f"Failed to fetch frame {label}: {e}")

    # Sort to maintain consistent order: thumb, 25%, 50%, 75%
    label_order = {v: i for i, (_, k) in enumerate(FRAME_URLS) for v in (k, _LABEL_DISPLAY[k])}
    results.sort(key=lambda x: label_order.get(x[1], 99))
    return results


def is_vertical_video(frames: list[tuple[np.ndarray, str]]) -> bool:
    """Detect vertical videos by checking for dark letterbox bars on left/right edges.

    YouTube thumbnails are always 4:3. Vertical videos get black side bars,
    detectable as dark edge strips (mean brightness < 40).
    """
    if not frames:
        return False

    _EDGE_WIDTH = 30
    _DARK_THRESHOLD = 55

    dark_count = 0
    for frame, _ in frames:
        h, w = frame.shape[:2]
        if w < _EDGE_WIDTH * 3:
            continue
        left_mean = np.mean(frame[:, :_EDGE_WIDTH, :])
        right_mean = np.mean(frame[:, -_EDGE_WIDTH:, :])
        # Center strip must be brighter than edges (real letterboxing, not a blank frame)
        center_start = w // 2 - _EDGE_WIDTH
        center_end = w // 2 + _EDGE_WIDTH
        center_mean = np.mean(frame[:, center_start:center_end, :])
        if left_mean < _DARK_THRESHOLD and right_mean < _DARK_THRESHOLD and center_mean > _DARK_THRESHOLD:
            dark_count += 1

    # At least half the frames must show letterboxing
    return dark_count >= max(2, len(frames) // 2)
