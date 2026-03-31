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

    Tries progressively larger windows centred on the seed.  Picks the
    *largest* window where the grid detector finds a valid 8×8 grid.
    Keeps trying even if intermediate sizes fail (the grid detector may
    fail at awkward aspect ratios where board edges are partially cropped
    but succeed again at the true board size).
    """
    from pipeline.overlay.grid_detector import detect_grid

    h, w = frame.shape[:2]
    sx, sy, sw, sh = seed_bbox
    cx = sx + sw // 2
    cy = sy + sh // 2

    best = seed_bbox

    # Try 125%, 150%, …, up to 6× the seed size.  Keep trying all
    # sizes — don't stop early because the grid detector can fail at
    # intermediate sizes and then succeed at the correct one.
    for multiplier_pct in range(125, 625, 25):
        size = int(sw * multiplier_pct / 100)
        if size > min(h, w):
            break

        ex = max(0, min(cx - size // 2, w - size))
        ey = max(0, min(cy - size // 2, h - size))
        size = min(size, w - ex, h - ey)

        crop = frame[ey : ey + size, ex : ex + size]
        # Apply light Gaussian blur to mitigate compression artifacts
        # (same as fast_overlay_check) before grid detection.
        blurred = cv2.GaussianBlur(crop, (3, 3), 0)
        # Skip uniform grid fallback — expansion crops include non-board
        # content, so a square crop should not auto-pass as a valid board.
        grid = detect_grid(blurred, allow_uniform=False)
        if grid is not None and len(grid.v_lines) == 9 and len(grid.h_lines) == 9:
            best = (ex, ey, size, size)

    # If the grid detector found the board at a larger size, use it to
    # compute a tighter bbox from the actual grid lines.
    if best != seed_bbox:
        ex, ey, size, _ = best
        crop = frame[ey : ey + size, ex : ex + size]
        blurred = cv2.GaussianBlur(crop, (3, 3), 0)
        grid = detect_grid(blurred, allow_uniform=False)
        if grid is not None and len(grid.v_lines) == 9 and len(grid.h_lines) == 9:
            gx = grid.v_lines[0]
            gy = grid.h_lines[0]
            gw = grid.v_lines[-1] - grid.v_lines[0]
            gh = grid.h_lines[-1] - grid.h_lines[0]
            board_size = max(gw, gh)
            bx = ex + gx
            by = ey + gy
            # Clamp
            bx = max(0, min(bx, w - board_size))
            by = max(0, min(by, h - board_size))
            board_size = min(board_size, w - bx, h - by)
            if board_size > sw:
                return (bx, by, board_size, board_size)

    return best


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


# Lightweight scan parameters for fast_overlay_check().
# Include small scales (0.20-0.30) for low-resolution video where
# compression artifacts prevent detection at board-sized windows.
# The small windows act as "seed" detections — grid regularity passes
# on small sub-regions where per-cell pixel counts are low enough that
# compression noise averages out.
FAST_SCAN_SCALES = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
FAST_SCAN_STEP_FRACTION = 0.25

# Only downscale frames significantly larger than this threshold.
# Low-res video (360p, 480p) must NOT be downscaled — compression
# artifacts already make grid regularity fragile at those resolutions.
FAST_CHECK_MAX_DIM = 810


def fast_overlay_check(frame: np.ndarray) -> OverlayDetection:
    """Fast overlay presence check — ~20x faster than detect_overlay_in_frame.

    Scans 8 scales (0.20-0.90) with larger step size.  Skips expansion
    and refinement.  Returns an approximate bbox suitable for segmentation
    (calibration refines it later).

    Automatically downscales frames larger than 810p for speed.  Applies
    a light Gaussian blur to mitigate video compression artifacts that
    inflate per-cell variance.  Returned bbox coordinates are scaled back
    to the original resolution.
    """
    h, w = frame.shape[:2]
    resolution = (w, h)

    # Downscale high-res frames for speed.  Don't downscale small frames
    # — compression artifacts are already worse at low resolution.
    scale_factor = 1.0
    if max(h, w) > FAST_CHECK_MAX_DIM:
        scale_factor = FAST_CHECK_MAX_DIM / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Light blur to smooth video compression artifacts (JPEG blocking,
    # color quantization).  This is critical for low-res video (360p/480p)
    # where per-cell variance can exceed the threshold due to compression
    # noise alone.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # (score, bbox, has_alternating_pattern)
    candidates: list[tuple[float, tuple[int, int, int, int], bool]] = []

    for scale in FAST_SCAN_SCALES:
        win_size = int(min(h, w) * scale)
        if win_size < 64:
            continue

        step = max(1, int(win_size * FAST_SCAN_STEP_FRACTION))
        scale_best_score = 0.0
        scale_best_bbox = None
        scale_best_pattern = False

        for y in range(0, h - win_size + 1, step):
            for x in range(0, w - win_size + 1, step):
                region = gray[y : y + win_size, x : x + win_size]
                regularity = compute_grid_regularity(region)

                if regularity > MIN_LOW_VARIANCE_RATIO:
                    has_pattern = check_alternating_pattern(region)
                    score = regularity + (0.2 if has_pattern else 0.0)

                    if score > scale_best_score:
                        scale_best_score = score
                        scale_best_bbox = (x, y, win_size, win_size)
                        scale_best_pattern = has_pattern

        if scale_best_bbox is not None:
            candidates.append((scale_best_score, scale_best_bbox, scale_best_pattern))

    # Prefer the largest detection that has the alternating light/dark
    # pattern.  This avoids false positives from camera views of physical
    # boards or solid-color UI panels that pass regularity but lack true
    # checkerboard alternation.  Fall back to the largest regularity-only
    # candidate if none have the pattern.
    best_score = 0.0
    best_bbox = None
    best_has_pattern = False
    for score, bbox, has_pattern in candidates:
        bbox_area = bbox[2] * bbox[3]
        best_area = best_bbox[2] * best_bbox[3] if best_bbox else 0

        # Prefer pattern candidates over non-pattern ones regardless of size.
        if has_pattern and not best_has_pattern:
            best_score = score
            best_bbox = bbox
            best_has_pattern = True
        elif has_pattern == best_has_pattern:
            if bbox_area > best_area or (bbox_area == best_area and score > best_score):
                best_score = score
                best_bbox = bbox
                best_has_pattern = has_pattern

    if best_bbox is not None and best_score > MIN_LOW_VARIANCE_RATIO:
        # Reject small detections that lack the alternating pattern —
        # likely false positives from physical boards or compression
        # artifacts at low resolution.
        if not best_has_pattern:
            best_fraction = best_bbox[2] / min(h, w)
            if best_fraction < 0.50:
                return OverlayDetection(found=False, frame_resolution=resolution)

        # Scale bbox back to original resolution if frame was downscaled
        if scale_factor < 1.0:
            inv = 1.0 / scale_factor
            best_bbox = (
                int(best_bbox[0] * inv),
                int(best_bbox[1] * inv),
                int(best_bbox[2] * inv),
                int(best_bbox[3] * inv),
            )
        return OverlayDetection(
            found=True,
            bbox=best_bbox,
            seed_bbox=best_bbox,
            score=best_score,
            frame_resolution=resolution,
        )

    return OverlayDetection(found=False, frame_resolution=resolution)


def detect_overlay_in_frame(frame: np.ndarray) -> OverlayDetection:
    """Detect a 2D chess board overlay in a video frame.

    Slides a square window across the frame at multiple scales,
    scoring each candidate region for rendered-board properties.
    Then expands the best detections outward to cover the full board
    (the initial scan may find a sub-region if the board has labels
    or coordinates around the edges).

    To handle frames with competing grid-like regions (e.g. camera views
    of physical boards alongside a rendered overlay), expansion is tried
    from the top candidates sorted by area.  The candidate whose expansion
    produces the largest valid board wins.
    """
    h, w = frame.shape[:2]
    resolution = (w, h)

    # Precompute grayscale once instead of per-window.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Collect all detections above threshold, then pick the best.
    # We prefer larger detections: a 900px board at 0.7 score is better
    # than a 216px sub-region at 0.93.  To achieve this we scan all scales
    # and keep the *largest* detection whose score exceeds the threshold.
    # Within the same scale, pick the highest score.
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []  # (score, bbox)

    for scale in SCAN_SCALES:
        win_size = int(min(h, w) * scale)
        if win_size < 64:
            continue

        step = max(1, int(win_size * SCAN_STEP_FRACTION))
        scale_best_score = 0.0
        scale_best_bbox = None

        for y in range(0, h - win_size + 1, step):
            for x in range(0, w - win_size + 1, step):
                region = gray[y : y + win_size, x : x + win_size]
                regularity = compute_grid_regularity(region)

                if regularity > MIN_LOW_VARIANCE_RATIO:
                    has_pattern = check_alternating_pattern(region)
                    score = regularity + (0.2 if has_pattern else 0.0)

                    if score > scale_best_score:
                        scale_best_score = score
                        scale_best_bbox = (x, y, win_size, win_size)

        if scale_best_bbox is not None:
            candidates.append((scale_best_score, scale_best_bbox))

    if not candidates:
        return OverlayDetection(found=False, frame_resolution=resolution)

    # Build a diverse set of seeds to try expansion from:
    # - Top 3 by window area (may include large but wrong candidates)
    # - Top 1 by score (highest confidence, often on the actual board)
    # The correct seed will expand to the largest valid board; wrong
    # seeds (camera views, UI elements) won't expand well because the
    # grid detector won't find real grid lines at larger sizes.
    by_area = sorted(candidates, key=lambda c: (c[1][2] * c[1][3], c[0]), reverse=True)
    by_score = sorted(candidates, key=lambda c: c[0], reverse=True)

    seeds_to_try: list[tuple[float, tuple[int, int, int, int]]] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for c in by_area[:3]:
        if c[1] not in seen_bboxes:
            seeds_to_try.append(c)
            seen_bboxes.add(c[1])
    # Always include the highest-scoring candidate if not already present.
    if by_score and by_score[0][1] not in seen_bboxes:
        seeds_to_try.append(by_score[0])

    best_expanded: tuple[int, int, int, int] | None = None
    best_expanded_area = 0
    best_seed_score = 0.0
    best_seed_bbox: tuple[int, int, int, int] | None = None

    for score, bbox in seeds_to_try:
        expanded = _expand_bbox(frame, bbox)
        exp_area = expanded[2] * expanded[3]
        if exp_area > best_expanded_area:
            best_expanded_area = exp_area
            best_expanded = expanded
            best_seed_score = score
            best_seed_bbox = bbox

    if best_expanded is not None and best_seed_bbox is not None:
        # If a small seed didn't expand AND lacks the alternating pattern,
        # the detection is likely a false positive (e.g. physical board at
        # low resolution).  Legitimate rendered overlays have the
        # alternating light/dark checkerboard pattern; physical boards
        # at low resolution usually don't after compression.
        gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        bx, by, bw, bh = best_expanded
        seed_has_pattern = check_alternating_pattern(
            gray_check[by : by + bh, bx : bx + bw]
        )
        seed_area = best_seed_bbox[2] * best_seed_bbox[3]
        seed_fraction = best_seed_bbox[2] / min(h, w)
        if (
            not seed_has_pattern
            and seed_fraction < 0.50
            and best_expanded_area <= seed_area * 1.2
        ):
            return OverlayDetection(found=False, frame_resolution=resolution)

        # Fine-tune alignment so the 8x8 grid lines up precisely.
        refined = _refine_alignment(frame, best_expanded)
        return OverlayDetection(
            found=True,
            bbox=refined,
            seed_bbox=best_seed_bbox,
            score=best_seed_score,
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
