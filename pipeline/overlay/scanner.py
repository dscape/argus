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
MAX_BOARD_FRACTION = 0.95

# Step size for sliding window (fraction of window size).
SCAN_STEP_FRACTION = 0.15

# Scales to try for the sliding window (fraction of frame height).
# Many tournament overlays occupy 50-95% of frame height, so we scan
# all the way up to 0.95.
SCAN_SCALES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


@dataclass
class OverlayDetection:
    """Result of overlay detection on a single frame."""

    found: bool
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h (expanded)
    seed_bbox: tuple[int, int, int, int] | None = None  # x, y, w, h (initial seed before expansion)
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
    margin_y = max(1, cell_h // 6)
    margin_x = max(1, cell_w // 6)
    inner_h = cell_h - 2 * margin_y
    inner_w = cell_w - 2 * margin_x

    if inner_h <= 0 or inner_w <= 0:
        return 0.0

    # Reshape into (8, cell_h, 8, cell_w) then transpose to (8, 8, cell_h, cell_w)
    trimmed = gray[: cell_h * 8, : cell_w * 8]
    grid = trimmed.reshape(8, cell_h, 8, cell_w).transpose(0, 2, 1, 3)

    # Apply uniform margin to each cell, flatten spatial dims, compute variance
    inner = grid[:, :, margin_y : cell_h - margin_y, margin_x : cell_w - margin_x]
    variances = inner.reshape(8, 8, -1).astype(np.float64).var(axis=2)

    return int(np.sum(variances < MAX_RENDERED_SQUARE_VARIANCE)) / 64.0


def check_alternating_pattern(region: np.ndarray) -> bool:
    """Check if a region has an alternating light/dark square pattern.

    Computes mean brightness per cell in the 8x8 grid and checks that
    adjacent cells have consistently different brightness.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    h, w = gray.shape
    cell_h = h // 8
    cell_w = w // 8

    if cell_h == 0 or cell_w == 0:
        return False

    # Compute per-cell means via reshape
    trimmed = gray[: cell_h * 8, : cell_w * 8]
    means = (
        trimmed.reshape(8, cell_h, 8, cell_w)
        .transpose(0, 2, 1, 3)
        .reshape(8, 8, -1)
        .mean(axis=2)
    )

    # Check horizontal and vertical alternation with vectorized diffs
    h_diffs = np.abs(means[:, :-1] - means[:, 1:])  # (8, 7)
    v_diffs = np.abs(means[:-1, :] - means[1:, :])  # (7, 8)
    alternation_count = int(np.sum(h_diffs > 15) + np.sum(v_diffs > 15))
    total_pairs = 8 * 7 + 7 * 8  # 112

    # Rendered boards won't have perfect alternation everywhere (pieces change
    # cell brightness), but should have it in most empty cells.
    return alternation_count / total_pairs > 0.35


def _expand_bbox(
    frame: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Expand a detected overlay bbox to cover the full board.

    The scanner finds a small high-confidence seed inside the board.
    This estimates the square size from the seed region and extrapolates
    to the full 8x8 board.

    Strategy: the seed's 8x8 grid gives us the approximate square size.
    From that we infer the full board size (8 * square_size) and position
    (align to the nearest grid boundary).
    """
    h, w = frame.shape[:2]
    sx, sy, sw, sh = seed_bbox

    # The seed was scored as a valid 8x8 grid, so each cell is sw/8.
    cell_size = sw / 8.0

    # Sample colors from the seed to find the board's light/dark palette.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Expand outward from the seed center. Walk left/right/up/down along
    # the board's grid pattern. At each step, check that the alternating
    # light/dark color pattern continues.
    cx = sx + sw // 2
    cy = sy + sh // 2

    # Measure light and dark square mean brightness from the seed.
    seed_gray = gray[sy : sy + sh, sx : sx + sw]
    cs = int(cell_size)
    light_vals, dark_vals = [], []
    for r in range(8):
        for c in range(8):
            y1, y2 = r * cs, (r + 1) * cs
            x1, x2 = c * cs, (c + 1) * cs
            cell = seed_gray[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            m = float(np.mean(cell))
            if (r + c) % 2 == 0:
                light_vals.append(m)
            else:
                dark_vals.append(m)

    if not light_vals or not dark_vals:
        return seed_bbox

    light_mean = np.median(light_vals)
    dark_mean = np.median(dark_vals)
    # Board squares should have a clear contrast between light and dark.
    if abs(light_mean - dark_mean) < 20:
        return seed_bbox

    mid_brightness = (light_mean + dark_mean) / 2
    tolerance = abs(light_mean - dark_mean) * 0.8

    def _is_board_pixel(px_y: int, px_x: int) -> bool:
        """Check if a pixel's brightness is within the board's color range."""
        if px_y < 0 or px_y >= h or px_x < 0 or px_x >= w:
            return False
        val = float(gray[px_y, px_x])
        return abs(val - light_mean) < tolerance or abs(val - dark_mean) < tolerance

    # Expand in each direction by scanning columns/rows.
    # Walk left from seed left edge until we hit non-board pixels.
    def _scan_edge(start: int, delta: int, axis: str, limit: int) -> int:
        """Scan outward from start in direction delta until board pattern stops."""
        pos = start
        consecutive_misses = 0
        while 0 <= pos + delta <= limit:
            pos += delta
            # Sample several points along this line
            hits = 0
            samples = 10
            for i in range(samples):
                if axis == "x":
                    py = sy + int(sh * i / samples)
                    px = pos
                else:
                    py = pos
                    px = sx + int(sw * i / samples)
                if _is_board_pixel(py, px):
                    hits += 1
            if hits < samples * 0.4:
                consecutive_misses += 1
                if consecutive_misses > int(cell_size * 0.5):
                    return pos - delta * consecutive_misses
            else:
                consecutive_misses = 0
        return pos

    left = _scan_edge(sx, -1, "x", w - 1)
    right = _scan_edge(sx + sw, 1, "x", w - 1)
    top = _scan_edge(sy, -1, "y", h - 1)
    bottom = _scan_edge(sy + sh, 1, "y", h - 1)

    # The board is square, so use the larger dimension.
    bw = right - left
    bh = bottom - top
    board_size = max(bw, bh)

    # Keep it square and centered on the detected region.
    bcx = (left + right) // 2
    bcy = (top + bottom) // 2
    half = board_size // 2
    ex = max(0, min(bcx - half, w - board_size))
    ey = max(0, min(bcy - half, h - board_size))
    board_size = min(board_size, w - ex, h - ey)

    # Only use the expanded bbox if it's meaningfully larger than the seed.
    if board_size < sw * 1.3:
        return seed_bbox

    # Trim uniform-color borders (rank/file label strips, separator bars).
    # Walk inward from each edge, skipping columns/rows with near-zero variance.
    expanded = frame[ey : ey + board_size, ex : ex + board_size]
    exp_gray = cv2.cvtColor(expanded, cv2.COLOR_BGR2GRAY) if len(expanded.shape) == 3 else expanded
    trim_l = trim_t = 0
    trim_r = trim_b = 0
    bs = board_size

    # Trim left
    for col_x in range(bs // 8):
        col = exp_gray[bs // 4 : 3 * bs // 4, col_x]
        if float(np.var(col)) < 100:
            trim_l = col_x + 1
        else:
            break

    # Trim right
    for col_x in range(bs - 1, bs - bs // 8, -1):
        col = exp_gray[bs // 4 : 3 * bs // 4, col_x]
        if float(np.var(col)) < 100:
            trim_r = bs - col_x
        else:
            break

    # Trim top
    for row_y in range(bs // 8):
        row = exp_gray[row_y, bs // 4 : 3 * bs // 4]
        if float(np.var(row)) < 100:
            trim_t = row_y + 1
        else:
            break

    # Trim bottom
    for row_y in range(bs - 1, bs - bs // 8, -1):
        row = exp_gray[row_y, bs // 4 : 3 * bs // 4]
        if float(np.var(row)) < 100:
            trim_b = bs - row_y
        else:
            break

    # Apply trim, keep square by using the max trim and adjusting center
    total_trim = max(trim_l + trim_r, trim_t + trim_b)
    if total_trim > 0 and total_trim < bs // 4:
        new_ex = ex + trim_l
        new_ey = ey + trim_t
        new_w = board_size - trim_l - trim_r
        new_h = board_size - trim_t - trim_b
        # Keep it square
        new_size = min(new_w, new_h)
        return (new_ex, new_ey, new_size, new_size)

    return (ex, ey, board_size, board_size)


def _refine_alignment(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    max_shift: int = 12,
) -> tuple[int, int, int, int]:
    """Fine-tune bbox position by maximizing grid regularity.

    The expansion step can leave the bbox a few pixels off from the true
    8x8 grid. This tries small shifts in each direction and picks the
    position that best aligns with the rendered board grid.

    Uses a precomputed grayscale frame to avoid repeated BGR->gray conversion.
    """
    x, y, w, h = bbox
    fh, fw = frame.shape[:2]

    # Precompute grayscale once — compute_grid_regularity skips cvtColor for 2D input
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    best_score = compute_grid_regularity(gray[y : y + h, x : x + w])
    best = bbox

    step = 4
    for dx in range(-max_shift, max_shift + 1, step):
        for dy in range(-max_shift, max_shift + 1, step):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx + w > fw or ny + h > fh:
                continue
            score = compute_grid_regularity(gray[ny : ny + h, nx : nx + w])
            if score > best_score:
                best_score = score
                best = (nx, ny, w, h)

    return best


def detect_overlay_in_frame(frame: np.ndarray) -> OverlayDetection:
    """Detect a 2D chess board overlay in a video frame.

    Slides a square window across the frame at multiple scales,
    scoring each candidate region for rendered-board properties.
    Then expands the best detection outward to cover the full board
    (the initial scan may find a sub-region if the board has labels
    or coordinates around the edges).
    """
    h, w = frame.shape[:2]
    resolution = (w, h)

    # Precompute grayscale once instead of per-window.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    best_score = 0.0
    best_bbox = None
    found_early = False

    for scale in SCAN_SCALES:
        if found_early:
            break

        win_size = int(min(h, w) * scale)
        if win_size < 64:
            continue

        step = max(1, int(win_size * SCAN_STEP_FRACTION))

        for y in range(0, h - win_size + 1, step):
            if found_early:
                break
            for x in range(0, w - win_size + 1, step):
                region = gray[y : y + win_size, x : x + win_size]
                regularity = compute_grid_regularity(region)

                if regularity > MIN_LOW_VARIANCE_RATIO:
                    has_pattern = check_alternating_pattern(region)
                    score = regularity + (0.2 if has_pattern else 0.0)

                    if score > best_score:
                        best_score = score
                        best_bbox = (x, y, win_size, win_size)

                        if best_score > 0.9:
                            found_early = True
                            break

    if best_bbox is not None and best_score > MIN_LOW_VARIANCE_RATIO:
        # Expand the seed detection to cover the full board.
        expanded = _expand_bbox(frame, best_bbox)
        # Fine-tune alignment so the 8x8 grid lines up precisely.
        expanded = _refine_alignment(frame, expanded)
        return OverlayDetection(
            found=True,
            bbox=expanded,
            seed_bbox=best_bbox,
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
