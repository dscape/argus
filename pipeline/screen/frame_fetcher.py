"""Fetch YouTube auto-generated frame thumbnails.

YouTube serves 4 auto-generated thumbnails per video at known URLs, with no
API quota cost. These are used for screening inspection and AI classification.

Images are cached on disk under data/frame_cache/{video_id}/ so that
re-extraction and re-training never need to re-download from YouTube.

Resolution note: YouTube thumbnails max out at 1280x720 (maxresdefault).
Overlay detection is tuned for 1920x1080 (native video resolution). Use
``fetch_overlay_frames_fullres()`` with yt-dlp to get 1920x1080 frames
for overlay detection and test fixtures.
"""

import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,15}$")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAME_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "screening", "dataset", "frames")
OVERLAY_FRAMES_DIR = os.path.join(_PROJECT_ROOT, "data", "overlay", "dataset", "frames")

# YouTube auto-generated frame URLs (publicly accessible, no API quota).
# hq variants serve 480x360 (vs 120x90 for plain 1/2/3.jpg).
# 0.jpg = default thumbnail (480x360), hq1/hq2/hq3 ~ 25%/50%/75% (480x360).
FRAME_URLS = [
    ("https://img.youtube.com/vi/{video_id}/0.jpg", "thumb"),
    ("https://img.youtube.com/vi/{video_id}/hq1.jpg", "25pct"),
    ("https://img.youtube.com/vi/{video_id}/hq2.jpg", "50pct"),
    ("https://img.youtube.com/vi/{video_id}/hq3.jpg", "75pct"),
]

# Higher-resolution default thumbnail URLs (tried in order, may 404).
# maxresdefault = 1280x720, sddefault = 640x480.
_HIRES_THUMB_URLS = [
    ("https://img.youtube.com/vi/{video_id}/maxresdefault.jpg", "thumb_hires"),
    ("https://img.youtube.com/vi/{video_id}/sddefault.jpg", "thumb_sd"),
]

# Mapping for backward compatibility with old label format
_LABEL_DISPLAY = {
    "thumb": "thumb", "25pct": "25%", "50pct": "50%", "75pct": "75%",
    "thumb_hires": "thumb_hires", "thumb_sd": "thumb_sd",
}

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


def fetch_hires_thumb(video_id: str) -> tuple[np.ndarray, str] | None:
    """Try to fetch a higher-resolution default thumbnail (1280x720 or 640x480).

    Checks cache first, then tries maxresdefault (1280x720) and sddefault
    (640x480) in order.  Returns ``(frame_bgr, label)`` or ``None`` if neither
    is available.  The result is cached to disk so subsequent calls are free.
    """
    if not _VIDEO_ID_RE.match(video_id):
        return None

    # Check cache for either hi-res variant
    for _, label in _HIRES_THUMB_URLS:
        cached = _load_cached_frame(video_id, label)
        if cached is not None:
            return (cached, label)

    # Download: try each URL in order (maxres first, then sd)
    for url_template, label in _HIRES_THUMB_URLS:
        url = url_template.format(video_id=video_id)
        frame = fetch_single_frame(url)
        if frame is not None:
            _save_frame_to_cache(video_id, label, frame)
            return (frame, label)

    return None


def fetch_youtube_frames(video_id: str) -> list[tuple[np.ndarray, str]]:
    """Fetch all YouTube auto-generated frames, using disk cache when available.

    Tries higher-resolution thumbnails first (maxresdefault 1280x720, sddefault
    640x480), then the standard 4 frames at 480x360.

    Returns list of (frame_bgr, display_label) tuples sorted in canonical order
    (hi-res first, then thumb, 25%, 50%, 75%).
    Frames are cached under data/frame_cache/{video_id}/ on first fetch.
    """
    if not _VIDEO_ID_RE.match(video_id):
        logger.warning(f"Invalid video_id format: {video_id!r}")
        return []

    results: list[tuple[np.ndarray, str]] = []
    to_fetch: list[tuple[str, str]] = []  # (url, label) pairs that need downloading

    # Try hi-res thumbnail first
    hires = fetch_hires_thumb(video_id)
    if hires is not None:
        results.append((hires[0], _LABEL_DISPLAY[hires[1]]))

    # First pass: check disk cache for standard frames
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

    # Sort: hi-res first, then standard order (thumb, 25%, 50%, 75%)
    _all_urls = _HIRES_THUMB_URLS + FRAME_URLS
    label_order = {v: i for i, (_, k) in enumerate(_all_urls) for v in (k, _LABEL_DISPLAY[k])}
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


# YouTube auto-generated frame URLs at 25%/50%/75% of video duration.
# maxres variants are 1280x720, hq variants are 480x360.
_OVERLAY_FRAME_URLS = [
    ("maxres1", "hq1", "25pct"),
    ("maxres2", "hq2", "50pct"),
    ("maxres3", "hq3", "75pct"),
]


def fetch_overlay_frames(
    video_id: str, hires: bool = True
) -> list[tuple[str, int, int]]:
    """Fetch 25/50/75% frames for a video at the requested resolution.

    When *hires* is True, tries maxres (1280x720) first, falls back to hq
    (480x360), and saves to ``OVERLAY_FRAMES_DIR``.  When False, fetches hq
    (480x360) only and saves to ``FRAME_CACHE_DIR`` (screening frames).

    Returns list of ``(label, width, height)`` for successfully fetched frames.
    Skips frames that are already cached on disk.
    """
    if not _VIDEO_ID_RE.match(video_id):
        return []

    output_dir = OVERLAY_FRAMES_DIR if hires else FRAME_CACHE_DIR
    results: list[tuple[str, int, int]] = []

    for hires_name, fallback_name, label in _OVERLAY_FRAME_URLS:
        output_path = os.path.join(output_dir, video_id, f"{label}.jpg")

        # Check cache
        if os.path.exists(output_path):
            frame = cv2.imread(output_path)
            if frame is not None and frame.shape[1] >= _MIN_FRAME_WIDTH:
                h, w = frame.shape[:2]
                results.append((label, w, h))
                continue

        # Download
        frame = None
        names = [hires_name, fallback_name] if hires else [fallback_name]
        for name in names:
            url = f"https://img.youtube.com/vi/{video_id}/{name}.jpg"
            frame = fetch_single_frame(url)
            if frame is not None:
                break

        if frame is None:
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = frame.shape[:2]
        results.append((label, w, h))

    return results


# ── Full-resolution frame extraction via yt-dlp ──────────────────────

# Overlay detection is tuned for 1920x1080 input. YouTube thumbnails cap at
# 1280x720, so we use yt-dlp to download short video sections and extract
# frames at native resolution.
OVERLAY_FRAME_RESOLUTION = (1920, 1080)

_YT_DLP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".venv", "bin", "yt-dlp",
)

# yt-dlp --download-sections requires ffmpeg. Ensure common install locations
# (Homebrew, etc.) are on PATH even if the parent shell doesn't include them.
_SUBPROCESS_ENV = {**os.environ}
for _extra in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _extra not in _SUBPROCESS_ENV.get("PATH", ""):
        _SUBPROCESS_ENV["PATH"] = _extra + ":" + _SUBPROCESS_ENV.get("PATH", "")

def get_video_duration(video_id: str) -> int:
    """Get video duration in seconds via yt-dlp metadata query."""
    if not _VIDEO_ID_RE.match(video_id) or not os.path.exists(_YT_DLP):
        return 0
    try:
        result = subprocess.run(
            [_YT_DLP, "--print", "duration", "--no-warnings",
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=30, check=False,
            env=_SUBPROCESS_ENV,
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except (ValueError, subprocess.TimeoutExpired):
        return 0


_OVERLAY_PERCENTAGES = [
    (0.25, "25pct"),
    (0.50, "50pct"),
    (0.75, "75pct"),
]


def _extract_frame_from_video(video_path: str) -> np.ndarray | None:
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _find_section_file(tmpdir: str) -> str | None:
    """Find the downloaded section file (yt-dlp may append format extension)."""
    import glob as glob_mod
    for pattern in ["section.*", "section*"]:
        matches = glob_mod.glob(os.path.join(tmpdir, pattern))
        if matches:
            return matches[0]
    return None


def fetch_overlay_frames_fullres(
    video_id: str, duration: int
) -> list[tuple[str, int, int]]:
    """Fetch 25/50/75% frames at native resolution (1920x1080) via yt-dlp.

    Downloads 2-second video sections at each timestamp and extracts the first
    frame at native resolution. Requires yt-dlp at ``.venv/bin/yt-dlp``.

    Args:
        video_id: YouTube video ID.
        duration: Video duration in seconds (used to compute timestamps).

    Returns:
        List of ``(label, width, height)`` for successfully fetched frames.
    """
    if not _VIDEO_ID_RE.match(video_id):
        return []

    if not os.path.exists(_YT_DLP):
        logger.error("yt-dlp not found at %s", _YT_DLP)
        return []

    results: list[tuple[str, int, int]] = []

    for pct, label in _OVERLAY_PERCENTAGES:
        output_path = os.path.join(OVERLAY_FRAMES_DIR, video_id, f"{label}.jpg")

        # Check cache
        if os.path.exists(output_path):
            frame = cv2.imread(output_path)
            if frame is not None and frame.shape[1] >= _MIN_FRAME_WIDTH:
                h, w = frame.shape[:2]
                results.append((label, w, h))
                logger.debug("Cached: %s/%s (%dx%d)", video_id, label, w, h)
                continue

        ts = int(duration * pct)
        url = f"https://www.youtube.com/watch?v={video_id}"

        with tempfile.TemporaryDirectory(prefix="argus_overlay_") as tmpdir:
            section_path = os.path.join(tmpdir, "section.mp4")

            # Download 2-second section at best quality (720p+)
            result = subprocess.run(
                [
                    _YT_DLP,
                    "--download-sections", f"*{ts}-{ts + 2}",
                    "-f", "bestvideo[height>=720][ext=mp4]/bestvideo[ext=mp4]/best[ext=mp4]/best",
                    "-o", section_path,
                    "--no-warnings", "--quiet",
                    url,
                ],
                capture_output=True, timeout=120, check=False,
                env=_SUBPROCESS_ENV,
            )

            if result.returncode != 0 or not os.path.exists(section_path):
                # Retry without format filter
                subprocess.run(
                    [
                        _YT_DLP,
                        "--download-sections", f"*{ts}-{ts + 2}",
                        "-f", "best",
                        "-o", section_path,
                        "--no-warnings", "--quiet",
                        url,
                    ],
                    capture_output=True, timeout=120, check=False,
                    env=_SUBPROCESS_ENV,
                )

            # yt-dlp may use a different extension
            actual_path = (
                section_path if os.path.exists(section_path)
                else _find_section_file(tmpdir)
            )
            if actual_path is None:
                logger.warning("Failed to download %s at t=%ds", video_id, ts)
                continue

            frame = _extract_frame_from_video(actual_path)
            if frame is None:
                logger.warning("Failed to extract frame from %s at t=%ds", video_id, ts)
                continue

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = frame.shape[:2]
            results.append((label, w, h))
            logger.info("Fetched %s/%s (%dx%d)", video_id, label, w, h)

    return results
