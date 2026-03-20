"""Screen crawled videos for 2D chess board overlay presence.

Rendered 2D boards (from lichess, chess.com, etc.) have distinctive properties:
- Perfect 8x8 grid of alternating-color squares
- Near-zero intra-square pixel variance (solid fills)
- Sharp boundaries between squares

This module samples 2-3 frames from each video and checks for these properties,
allowing cheap bulk screening of all crawled videos.
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.db.connection import get_conn

logger = logging.getLogger(__name__)

# Rendered board squares have very low pixel variance (solid color fills).
# OTB boards have much higher variance from lighting, wood grain, reflections.
MAX_RENDERED_SQUARE_VARIANCE = 25.0

# Minimum ratio of low-variance cells to consider a region as a rendered board.
# 8x8 = 64 cells; we require at least ~50 to be low-variance (pieces add variance).
MIN_LOW_VARIANCE_RATIO = 0.55

# Timestamps (seconds) to sample frames from each video.
# Skip first 30s to avoid intros.
SAMPLE_TIMESTAMPS = [30, 120, 300]

# Minimum board size as fraction of frame dimension.
MIN_BOARD_FRACTION = 0.10
MAX_BOARD_FRACTION = 0.60

# Step size for sliding window (fraction of window size).
SCAN_STEP_FRACTION = 0.15

# Scales to try for the sliding window (fraction of frame height).
SCAN_SCALES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


@dataclass
class OverlayDetection:
    """Result of overlay detection on a single frame."""

    found: bool
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h
    score: float = 0.0
    frame_resolution: tuple[int, int] | None = None  # width, height


def compute_grid_regularity(region: np.ndarray) -> float:
    """Score how 'rendered' a candidate board region looks.

    Divides the region into an 8x8 grid and computes per-cell pixel variance.
    Rendered boards have very low variance (solid fills); real boards have high
    variance from lighting, texture, etc.

    Returns the fraction of cells with variance below the threshold.
    """
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    h, w = gray.shape

    if h < 32 or w < 32:
        return 0.0

    cell_h = h // 8
    cell_w = w // 8
    low_variance_count = 0

    for row in range(8):
        for col in range(8):
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            cell = gray[y1:y2, x1:x2]

            # Shrink slightly to avoid edge effects between squares
            margin_y = max(1, cell_h // 6)
            margin_x = max(1, cell_w // 6)
            inner = cell[margin_y:-margin_y, margin_x:-margin_x]

            if inner.size == 0:
                continue

            variance = float(np.var(inner))
            if variance < MAX_RENDERED_SQUARE_VARIANCE:
                low_variance_count += 1

    return low_variance_count / 64.0


def check_alternating_pattern(region: np.ndarray) -> bool:
    """Check if a region has an alternating light/dark square pattern.

    Computes mean brightness per cell in the 8x8 grid and checks that
    adjacent cells have consistently different brightness.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    h, w = gray.shape
    cell_h = h // 8
    cell_w = w // 8

    means = np.zeros((8, 8))
    for row in range(8):
        for col in range(8):
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            means[row, col] = np.mean(gray[y1:y2, x1:x2])

    # Check horizontal alternation: adjacent cells should differ
    alternation_count = 0
    total_pairs = 0
    for row in range(8):
        for col in range(7):
            diff = abs(means[row, col] - means[row, col + 1])
            total_pairs += 1
            if diff > 15:  # Meaningful brightness difference
                alternation_count += 1

    # Also check vertical
    for row in range(7):
        for col in range(8):
            diff = abs(means[row, col] - means[row + 1, col])
            total_pairs += 1
            if diff > 15:
                alternation_count += 1

    # Rendered boards won't have perfect alternation everywhere (pieces change
    # cell brightness), but should have it in most empty cells.
    return alternation_count / total_pairs > 0.35 if total_pairs > 0 else False


def detect_overlay_in_frame(frame: np.ndarray) -> OverlayDetection:
    """Detect a 2D chess board overlay in a video frame.

    Slides a square window across the frame at multiple scales,
    scoring each candidate region for rendered-board properties.
    """
    h, w = frame.shape[:2]
    resolution = (w, h)

    best_score = 0.0
    best_bbox = None

    for scale in SCAN_SCALES:
        win_size = int(min(h, w) * scale)
        if win_size < 64:
            continue

        step = max(1, int(win_size * SCAN_STEP_FRACTION))

        for y in range(0, h - win_size + 1, step):
            for x in range(0, w - win_size + 1, step):
                region = frame[y : y + win_size, x : x + win_size]
                regularity = compute_grid_regularity(region)

                if regularity > MIN_LOW_VARIANCE_RATIO:
                    has_pattern = check_alternating_pattern(region)
                    score = regularity + (0.2 if has_pattern else 0.0)

                    if score > best_score:
                        best_score = score
                        best_bbox = (x, y, win_size, win_size)

    if best_bbox is not None and best_score > MIN_LOW_VARIANCE_RATIO:
        return OverlayDetection(
            found=True,
            bbox=best_bbox,
            score=best_score,
            frame_resolution=resolution,
        )

    return OverlayDetection(found=False, frame_resolution=resolution)


def extract_frames_from_video(
    video_url_or_path: str,
    timestamps: list[int] | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Extract specific frames from a video using yt-dlp + ffmpeg.

    For YouTube URLs, downloads a short section around each timestamp.
    For local files, extracts frames directly with ffmpeg.

    Returns list of paths to extracted frame images.
    """
    if timestamps is None:
        timestamps = SAMPLE_TIMESTAMPS

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="argus_scan_")

    frame_paths = []
    is_url = video_url_or_path.startswith(("http://", "https://"))

    if is_url:
        # Download short sections via yt-dlp
        for i, ts in enumerate(timestamps):
            section_path = os.path.join(output_dir, f"section_{i}.mp4")
            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")

            try:
                # Download 2 seconds around the timestamp at lowest quality
                subprocess.run(
                    [
                        "yt-dlp",
                        "--download-sections", f"*{ts}-{ts + 2}",
                        "-f", "worst[ext=mp4]/worst",
                        "-o", section_path,
                        "--no-warnings",
                        "--quiet",
                        video_url_or_path,
                    ],
                    capture_output=True,
                    timeout=60,
                    check=False,
                )

                if os.path.exists(section_path):
                    # Extract first frame
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", section_path,
                            "-frames:v", "1",
                            "-q:v", "2",
                            frame_path,
                        ],
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )

                    if os.path.exists(frame_path):
                        frame_paths.append(frame_path)

                    # Clean up section
                    os.remove(section_path)

            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning(f"Failed to extract frame at {ts}s from {video_url_or_path}: {e}")
                continue
    else:
        # Local file: extract frames with ffmpeg
        for i, ts in enumerate(timestamps):
            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(ts),
                        "-i", video_url_or_path,
                        "-frames:v", "1",
                        "-q:v", "2",
                        frame_path,
                    ],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )

                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)

            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning(f"Failed to extract frame at {ts}s from {video_url_or_path}: {e}")
                continue

    return frame_paths


def scan_video(video_url_or_path: str) -> OverlayDetection:
    """Scan a single video for overlay presence.

    Extracts 2-3 frames and checks each for a 2D board overlay.
    Returns the best detection result.
    """
    frame_paths = extract_frames_from_video(video_url_or_path)

    if not frame_paths:
        logger.warning(f"Could not extract any frames from {video_url_or_path}")
        return OverlayDetection(found=False)

    best_detection = OverlayDetection(found=False)

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        detection = detect_overlay_in_frame(frame)
        if detection.found and detection.score > best_detection.score:
            best_detection = detection

    # Clean up extracted frames
    for path in frame_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    return best_detection


def scan_crawled_videos(
    channel_handle: str | None = None,
    limit: int | None = None,
):
    """Screen crawled videos for overlay presence and tag them in the DB.

    Processes videos that haven't been screened yet (layout_type IS NULL).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT video_id, channel_handle
                FROM youtube_videos
                WHERE layout_type IS NULL
            """
            params: list = []

            if channel_handle:
                query += " AND channel_handle = %s"
                params.append(channel_handle)

            query += " ORDER BY published_at DESC"

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            cur.execute(query, params)
            videos = cur.fetchall()

    if not videos:
        print("No unscreened videos found.")
        return

    print(f"Screening {len(videos)} videos for overlay presence...")
    overlay_count = 0
    otb_count = 0
    failed = 0

    for video_id, handle in videos:
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            detection = scan_video(url)

            layout_type = "overlay" if detection.found else "otb_only"

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE youtube_videos SET layout_type = %s WHERE video_id = %s",
                        (layout_type, video_id),
                    )
                    conn.commit()

            if detection.found:
                overlay_count += 1
                logger.info(
                    f"OVERLAY: {video_id} (score={detection.score:.2f}, "
                    f"bbox={detection.bbox})"
                )
            else:
                otb_count += 1

        except Exception as e:
            failed += 1
            logger.error(f"Failed to scan {video_id}: {e}")

    print(
        f"\nScan complete: {overlay_count} overlay, "
        f"{otb_count} OTB-only, {failed} failed"
    )
