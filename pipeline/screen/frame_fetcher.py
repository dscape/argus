"""Fetch YouTube auto-generated frame thumbnails.

YouTube serves 4 auto-generated thumbnails per video at known URLs, with no
API quota cost. These are used for screening inspection and AI classification.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# YouTube auto-generated frame URLs (publicly accessible, no API quota).
# hq variants serve 480x360 (vs 120x90 for plain 1/2/3.jpg).
# 0.jpg = default thumbnail (480x360), hq1/hq2/hq3 ~ 25%/50%/75% (480x360).
FRAME_URLS = [
    ("https://img.youtube.com/vi/{video_id}/0.jpg", "thumb"),
    ("https://img.youtube.com/vi/{video_id}/hq1.jpg", "25%"),
    ("https://img.youtube.com/vi/{video_id}/hq2.jpg", "50%"),
    ("https://img.youtube.com/vi/{video_id}/hq3.jpg", "75%"),
]

_MIN_FRAME_WIDTH = 100


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
    """Fetch all 4 YouTube auto-generated frames in parallel.

    Returns list of (frame_bgr, label) tuples sorted in canonical order.
    """
    results: list[tuple[np.ndarray, str]] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                fetch_single_frame, url_template.format(video_id=video_id)
            ): label
            for url_template, label in FRAME_URLS
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                frame = future.result()
                if frame is not None:
                    results.append((frame, label))
            except Exception as e:
                logger.warning(f"Failed to fetch frame {label}: {e}")

    # Sort to maintain consistent order: thumb, 25%, 50%, 75%
    label_order = {label: i for i, (_, label) in enumerate(FRAME_URLS)}
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
    _DARK_THRESHOLD = 40

    dark_count = 0
    for frame, _ in frames:
        h, w = frame.shape[:2]
        if w < _EDGE_WIDTH * 3:
            continue
        left_mean = np.mean(frame[:, :_EDGE_WIDTH, :])
        right_mean = np.mean(frame[:, -_EDGE_WIDTH:, :])
        if left_mean < _DARK_THRESHOLD and right_mean < _DARK_THRESHOLD:
            dark_count += 1

    # Majority of frames must show letterboxing
    return dark_count >= len(frames) // 2 + 1
