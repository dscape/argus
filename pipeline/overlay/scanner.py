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
# 8x8 = 64 cells; we require at least ~29 to be low-variance.  Piece-heavy
# mid-game positions (e.g. chess24 overlays with detailed graphics) can push
# per-cell variance above the threshold on ~29 occupied cells, so we need
# headroom.  False positives are controlled by check_alternating_pattern().
MIN_LOW_VARIANCE_RATIO = 0.45

# Maximum standard deviation of mean brightness across same-color
# checkerboard positions (among low-variance cells).  Real boards
# typically have std < 5; bodies and clothing have std > 20.
MAX_SAME_COLOR_STD = 15.0

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


def compute_alternation_strength(region: np.ndarray) -> tuple[float, float]:
    """Compute alternation fraction and mean contrast for an 8x8 grid.

    Returns (frac_alternating, avg_contrast).  Real overlay boards have
    frac >= 0.65 and contrast >= 30; false positives from UI elements,
    banners, or OTB footage score lower.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    h, w = gray.shape
    cell_h = h // 8
    cell_w = w // 8

    if cell_h == 0 or cell_w == 0:
        return 0.0, 0.0

    trimmed = gray[: cell_h * 8, : cell_w * 8]
    means = (
        trimmed.reshape(8, cell_h, 8, cell_w)
        .transpose(0, 2, 1, 3)
        .reshape(8, 8, -1)
        .mean(axis=2)
    )

    h_diffs = np.abs(means[:, :-1] - means[:, 1:])
    v_diffs = np.abs(means[:-1, :] - means[1:, :])
    all_diffs = np.concatenate([h_diffs.ravel(), v_diffs.ravel()])

    alternation_count = int(np.sum(all_diffs > 15))
    total_pairs = len(all_diffs)
    frac = alternation_count / total_pairs if total_pairs > 0 else 0.0
    avg_contrast = float(np.mean(all_diffs)) if len(all_diffs) > 0 else 0.0

    return frac, avg_contrast


# Thresholds for strong alternation (separates real boards from false positives).
MIN_ALTERNATION_FRAC = 0.65
MIN_ALTERNATION_CONTRAST = 30.0

# Relaxed thresholds for Phase 1 candidate generation.  Phase 2 validates
# at full resolution with checkerboard consistency, so Phase 1 can afford
# more permissive thresholds to avoid rejecting real boards due to grid
# misalignment, dark themes, or piece-heavy positions.
_P1_MIN_FRAC = 0.50
_P1_MIN_CONTRAST = 20.0

# Maximum std of cell brightness within each checkerboard group (light/dark).
# Real rendered boards typically have max_std <= 25 on empty boards and up to
# ~35 with many pieces.  OTB boards on vinyl surfaces score 38-52 due to 3D
# lighting and piece shadows.  Threshold of 35 catches most rendered boards
# while rejecting most OTB boards.
MAX_CHECKERBOARD_STD = 30.0

# Geometry fallback thresholds. These are intentionally stricter than the
# fast scan thresholds because this path can fire without a texture-based seed.
_GEOM_MIN_REGULARITY = 0.35
_GEOM_MIN_ALTERNATION_FRAC = 0.80
_GEOM_MIN_ALTERNATION_CONTRAST = 35.0
_GEOM_MIN_PERIODICITY = 1.20


def _checkerboard_std(gray: np.ndarray) -> float:
    """Compute max(light_std, dark_std) for the 8x8 grid of a region."""
    h, w = gray.shape
    cell_h = h // 8
    cell_w = w // 8
    if cell_h == 0 or cell_w == 0:
        return 999.0

    trimmed = gray[: cell_h * 8, : cell_w * 8]
    means = (
        trimmed.reshape(8, cell_h, 8, cell_w)
        .transpose(0, 2, 1, 3)
        .reshape(8, 8, -1)
        .mean(axis=2)
    )

    light = [means[r, c] for r in range(8) for c in range(8) if (r + c) % 2 == 0]
    dark = [means[r, c] for r in range(8) for c in range(8) if (r + c) % 2 == 1]

    return max(float(np.std(light)), float(np.std(dark)))


def check_checkerboard_consistency(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    """Check checkerboard consistency trying multiple sub-cell offsets.

    The coarse detection window may not align precisely with the actual
    board cells.  This tries offsets of ±half-cell in each direction
    to find the best alignment, passing if ANY offset achieves low
    checkerboard std.
    """
    x, y, w, h_box = bbox
    fh, fw = gray.shape[:2]
    cell = w // 16  # half-cell offset

    best_std = 999.0
    for dy in range(-cell, cell + 1, max(1, cell // 2)):
        for dx in range(-cell, cell + 1, max(1, cell // 2)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx + w > fw or ny + h_box > fh:
                continue
            s = _checkerboard_std(gray[ny: ny + h_box, nx: nx + w])
            if s < best_std:
                best_std = s

    return best_std <= MAX_CHECKERBOARD_STD


def _projection_comb_score(projection: np.ndarray, spacing: float) -> float:
    """Score how well a 1D projection matches a 9-line chessboard comb.

    Unlike FFT-based periodicity, this keeps phase information and directly
    rewards the expected internal board boundaries while penalizing energy in
    the middle of squares. Tight board crops often lose the outer edges, so we
    focus on the 7 interior lines rather than requiring all 9 boundaries.
    """
    axis_len = len(projection)
    if axis_len == 0 or spacing < 4:
        return 0.0

    sigma = max(spacing * 0.08, 0.8)
    smoothed = cv2.GaussianBlur(
        projection.reshape(1, -1).astype(np.float32),
        (0, 0),
        sigmaX=sigma,
    ).ravel()

    band_radius = max(1, int(round(spacing * 0.12)))
    phase_limit = max(1, int(round(spacing)))
    phase_step = max(1, phase_limit // 8)
    best_score = 0.0

    for phase in range(0, phase_limit, phase_step):
        line_vals: list[float] = []
        gap_vals: list[float] = []

        for i in range(1, 8):
            center = int(round(phase + i * spacing))
            if center < 0 or center >= axis_len:
                line_vals = []
                break
            lo = max(0, center - band_radius)
            hi = min(axis_len, center + band_radius + 1)
            line_vals.append(float(np.mean(smoothed[lo:hi])))

        if len(line_vals) != 7:
            continue

        for i in range(8):
            center = int(round(phase + (i + 0.5) * spacing))
            if center < 0 or center >= axis_len:
                gap_vals = []
                break
            lo = max(0, center - band_radius)
            hi = min(axis_len, center + band_radius + 1)
            gap_vals.append(float(np.mean(smoothed[lo:hi])))

        if len(gap_vals) != 8:
            continue

        line_mean = float(np.mean(line_vals))
        gap_mean = float(np.mean(gap_vals))
        if line_mean <= 0:
            continue

        uniformity = 1.0 - min(1.0, float(np.std(line_vals)) / max(line_mean, 1e-6))
        score = ((line_mean + 1.0) / (gap_mean + 1.0)) * (0.6 + 0.4 * uniformity)
        if score > best_score:
            best_score = score

    return best_score


def compute_axis_aligned_periodicity(region: np.ndarray) -> float:
    """Measure phase-aware x/y grid periodicity for an 8x8 board crop."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return 0.0

    gray_f = gray.astype(np.float32)
    sobel_x = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3))

    v_proj = np.mean(sobel_x, axis=0)
    h_proj = np.mean(sobel_y, axis=1)

    v_score = _projection_comb_score(v_proj, w / 8.0)
    h_score = _projection_comb_score(h_proj, h / 8.0)

    return min(v_score, h_score)


def _grid_result_to_bbox(
    grid,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Convert grid lines into a clamped square bounding box."""
    h, w = frame_shape[:2]
    gx1, gx2 = grid.v_lines[0], grid.v_lines[-1]
    gy1, gy2 = grid.h_lines[0], grid.h_lines[-1]

    board_w = max(1, gx2 - gx1)
    board_h = max(1, gy2 - gy1)
    board_size = max(board_w, board_h)
    gx = max(0, min(gx1, w - board_size))
    gy = max(0, min(gy1, h - board_size))
    board_size = min(board_size, w - gx, h - gy)

    return gx, gy, board_size, board_size


def _bbox_looks_like_overlay(
    frame: np.ndarray,
    gray_full: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    """Validate a geometry-only bbox using appearance and comb periodicity."""
    x, y, w, h = bbox
    if w < 64 or h < 64:
        return False

    region_gray = gray_full[y : y + h, x : x + w]
    if region_gray.size == 0:
        return False

    regularity = compute_grid_regularity(region_gray)
    frac, contrast = compute_alternation_strength(region_gray)
    periodicity = compute_axis_aligned_periodicity(region_gray)

    if (
        regularity < _GEOM_MIN_REGULARITY
        or frac < _GEOM_MIN_ALTERNATION_FRAC
        or contrast < _GEOM_MIN_ALTERNATION_CONTRAST
        or periodicity < _GEOM_MIN_PERIODICITY
    ):
        return False

    # Checkerboard/color consistency are useful extra evidence when the crop
    # includes the full widget, but some valid board-tight crops with pieces
    # do not pass them. Treat them as support, not as hard gates.
    if min(gray_full.shape[:2]) >= 720 and check_checkerboard_consistency(gray_full, bbox):
        return True

    return True


def check_color_consistency(region: np.ndarray) -> bool:
    """Check that low-variance cells at same checkerboard positions share consistent colors.

    On a rendered board, all empty light squares are the same color and all
    empty dark squares are the same color.  False positives (arms, clothing,
    UI panels) fail this because cells at matching checkerboard positions
    span a wide range of brightnesses.

    Only considers low-variance cells (likely empty squares) to avoid being
    thrown off by pieces.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    h, w = gray.shape
    cell_h = h // 8
    cell_w = w // 8

    if cell_h < 4 or cell_w < 4:
        return False

    margin_y = max(1, cell_h // 6)
    margin_x = max(1, cell_w // 6)
    inner_h = cell_h - 2 * margin_y
    inner_w = cell_w - 2 * margin_x

    if inner_h <= 0 or inner_w <= 0:
        return False

    trimmed = gray[: cell_h * 8, : cell_w * 8]
    grid = trimmed.reshape(8, cell_h, 8, cell_w).transpose(0, 2, 1, 3)
    inner = grid[:, :, margin_y : cell_h - margin_y, margin_x : cell_w - margin_x]
    flat = inner.reshape(8, 8, -1).astype(np.float64)
    variances = flat.var(axis=2)
    means = flat.mean(axis=2)

    # Collect means from low-variance cells, grouped by checkerboard position.
    # Exclude near-black cells (mean < 15) — these are typically background
    # or border pixels when the scan window extends beyond the actual board.
    light_mask = np.zeros((8, 8), dtype=bool)
    dark_mask = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for c in range(8):
            if variances[r, c] < MAX_RENDERED_SQUARE_VARIANCE and means[r, c] > 15.0:
                if (r + c) % 2 == 0:
                    light_mask[r, c] = True
                else:
                    dark_mask[r, c] = True

    light_means = means[light_mask]
    dark_means = means[dark_mask]

    # Need enough cells in each group to judge consistency.
    if len(light_means) < 4 or len(dark_means) < 4:
        return False

    light_std = float(np.std(light_means))
    dark_std = float(np.std(dark_means))

    return light_std < MAX_SAME_COLOR_STD and dark_std < MAX_SAME_COLOR_STD


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

    # Pass 1: coarse search (4px steps)
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

    # Pass 2: fine search (1px steps around coarse best)
    bx, by = best[0], best[1]
    fine_shift = step - 1
    for dx in range(-fine_shift, fine_shift + 1):
        for dy in range(-fine_shift, fine_shift + 1):
            nx, ny = bx + dx, by + dy
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
# NOTE: Overlay detection is tuned for 1920x1080 input (native YouTube).
# YouTube thumbnails (maxresdefault) are only 1280x720 — the smaller cell
# sizes shift variance/contrast distributions and reduce detection accuracy.
# Always use yt-dlp-extracted frames at 1920x1080 for reliable results.
FAST_CHECK_MAX_DIM = 810


def _expand_fast(
    gray: np.ndarray,
    seed: tuple[int, int, int, int],
    has_pattern: bool,
) -> tuple[int, int, int, int]:
    """Expand a seed bbox outward to find the full board boundary.

    Tries progressively larger square windows centred on the seed
    midpoint, and also shifted towards each edge.  Keeps the largest
    window that still passes the regularity + alternating pattern checks.

    Returns the expanded bbox in (x, y, w, h) format, or the original
    seed if no expansion improves it.
    """
    h, w = gray.shape[:2]
    sx, sy, sw, sh = seed

    cx = sx + sw // 2
    cy = sy + sh // 2

    best = seed
    best_area = sw * sh

    # Try multiple centres: seed centre + shifted towards each edge.
    # This handles boards that are near a frame edge where the seed
    # is off-centre relative to the true board.
    half = sw // 2
    centres = [
        (cx, cy),
        (cx - half, cy),
        (cx + half, cy),
        (cx, cy - half),
        (cx, cy + half),
    ]

    # Grow from 110% to 350% of seed size in 10% increments.
    for ocx, ocy in centres:
        for mult_pct in range(110, 350, 10):
            size = int(sw * mult_pct / 100)
            if size > max(h, w):
                break

            ex = max(0, min(ocx - size // 2, w - size))
            ey = max(0, min(ocy - size // 2, h - size))
            # Clamp size to fit in frame
            actual_w = min(size, w - ex)
            actual_h = min(size, h - ey)
            actual_size = min(actual_w, actual_h)
            if actual_size < 64:
                continue

            region = gray[ey : ey + actual_size, ex : ex + actual_size]
            reg = compute_grid_regularity(region)
            if reg < MIN_LOW_VARIANCE_RATIO:
                continue

            # Require alternating pattern to prevent expanding into
            # non-board areas (player panels, banners) that happen
            # to pass regularity.
            if has_pattern and not check_alternating_pattern(region):
                continue

            area = actual_size * actual_size
            if area > best_area:
                best = (ex, ey, actual_size, actual_size)
                best_area = area

    return best


def _refine_alternation(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], float, float]:
    """Refine bbox position to maximize alternation and grid alignment.

    Tries shifts of ±12px at 4px steps and size adjustments of ±10%.
    """
    x, y, w, h_box = bbox
    fh, fw = gray.shape[:2]

    def _score(nx: int, ny: int, nw: int, nh: int) -> tuple[float, float, float]:
        region = gray[ny : ny + nh, nx : nx + nw]
        frac, contrast = compute_alternation_strength(region)
        return frac * contrast, frac, contrast

    best_score, best_frac, best_contrast = _score(x, y, w, h_box)
    best = bbox

    # Position shifts
    for dy in range(-12, 13, 4):
        for dx in range(-12, 13, 4):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx + w > fw or ny + h_box > fh:
                continue
            score, frac, contrast = _score(nx, ny, w, h_box)
            if score > best_score:
                best_score, best_frac, best_contrast = score, frac, contrast
                best = (nx, ny, w, h_box)

    # Size adjustments (±10%)
    bx, by, bw, bh = best
    for size_pct in [90, 95, 105, 110]:
        ns = int(bw * size_pct / 100)
        ncx, ncy = bx + bw // 2, by + bh // 2
        nx = max(0, min(ncx - ns // 2, fw - ns))
        ny = max(0, min(ncy - ns // 2, fh - ns))
        if ns < 64 or nx + ns > fw or ny + ns > fh:
            continue
        score, frac, contrast = _score(nx, ny, ns, ns)
        if score > best_score:
            best_score, best_frac, best_contrast = score, frac, contrast
            best = (nx, ny, ns, ns)

    return best, best_frac, best_contrast


def fast_overlay_check(frame: np.ndarray) -> OverlayDetection:
    """Fast overlay presence check — under 100ms on 1080p.

    Two-phase approach:
    1. Scan at downscaled resolution (810p) for speed using alternation check
    2. Validate at full resolution with checkerboard consistency

    This combines the speed of downscaled scanning with the accuracy of
    full-resolution validation.  The checkerboard check prevents false
    positives from UI elements that pass the alternation threshold.
    """
    orig_h, orig_w = frame.shape[:2]
    resolution = (orig_w, orig_h)

    # Prepare full-res grayscale for validation.
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    gray_full = cv2.GaussianBlur(gray_full, (3, 3), 0)

    # Downscale for scan phase.
    scale_factor = 1.0
    if max(orig_h, orig_w) > FAST_CHECK_MAX_DIM:
        scale_factor = FAST_CHECK_MAX_DIM / max(orig_h, orig_w)
        small = cv2.resize(frame, (int(orig_w * scale_factor), int(orig_h * scale_factor)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame
    h, w = small.shape[:2] if len(small.shape) == 3 else small.shape

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Phase 1: Scan at downscaled resolution.
    # Two detection paths: alternation (all board types) + regularity (flat boards).
    #
    # Two-pass strategy for grid alignment:
    #   Pass 1: Coarse scan at normal step (25% of window = 2 cells).
    #           Collect "near-miss" windows that show some alternation but
    #           don't meet the full threshold (grid misalignment).
    #   Pass 2: Refine near-misses with half-cell offsets (4 trials each).
    #
    # This avoids the cost of offset trials on every window while still
    # recovering boards that the coarse scan misses due to alignment.
    candidates: list[tuple[float, int, tuple[int, int, int, int]]] = []

    # Lower bar for "promising" windows that deserve offset refinement.
    _NEAR_MISS_FRAC = 0.40
    _NEAR_MISS_CONTRAST = 15.0

    for scale in FAST_SCAN_SCALES:
        win_size = int(min(h, w) * scale)
        if win_size < 64:
            continue

        step = max(1, int(win_size * FAST_SCAN_STEP_FRACTION))
        half_cell = win_size // 16  # half of one cell width
        scale_best: tuple[float, tuple[int, int, int, int]] | None = None
        near_misses: list[tuple[float, int, int]] = []

        for y in range(0, h - win_size + 1, step):
            for x in range(0, w - win_size + 1, step):
                region = gray[y: y + win_size, x: x + win_size]

                # Path A: strong alternation pattern
                frac, contrast = compute_alternation_strength(region)
                if frac >= _P1_MIN_FRAC and contrast >= _P1_MIN_CONTRAST:
                    if scale_best is None or frac > scale_best[0]:
                        scale_best = (frac, (x, y, win_size, win_size))
                    continue

                # Path B: grid regularity with basic pattern check
                regularity = compute_grid_regularity(region)
                if regularity >= MIN_LOW_VARIANCE_RATIO and check_alternating_pattern(region):
                    if scale_best is None or regularity > scale_best[0]:
                        scale_best = (regularity, (x, y, win_size, win_size))
                    continue

                # Track near-misses for offset refinement
                if frac >= _NEAR_MISS_FRAC and contrast >= _NEAR_MISS_CONTRAST:
                    near_misses.append((frac, x, y))

        # Pass 2: refine top near-misses with half-cell offsets.
        if scale_best is None and near_misses:
            near_misses.sort(reverse=True)
            for _, nm_x, nm_y in near_misses[:5]:
                for dy in (0, half_cell, -half_cell):
                    for dx in (0, half_cell, -half_cell):
                        if dx == 0 and dy == 0:
                            continue
                        rx, ry = nm_x + dx, nm_y + dy
                        if rx < 0 or ry < 0 or rx + win_size > w or ry + win_size > h:
                            continue
                        region = gray[ry: ry + win_size, rx: rx + win_size]
                        frac, contrast = compute_alternation_strength(region)
                        if frac >= _P1_MIN_FRAC and contrast >= _P1_MIN_CONTRAST:
                            if scale_best is None or frac > scale_best[0]:
                                scale_best = (frac, (rx, ry, win_size, win_size))

        if scale_best is not None:
            f, b = scale_best
            candidates.append((f, b[2] * b[3], b))

    # Phase 2: Map to full resolution + validate with checkerboard.
    # Sort by score (desc) to try best-confidence candidates first.
    # Larger windows that include non-board content score lower, so
    # score-based sorting naturally prefers board-only detections.
    candidates.sort(key=lambda c: c[0], reverse=True)
    inv = 1.0 / scale_factor if scale_factor < 1.0 else 1.0

    for _, _, small_bbox in candidates:
        # Map bbox back to full resolution.
        if scale_factor < 1.0:
            full_bbox = (
                int(small_bbox[0] * inv),
                int(small_bbox[1] * inv),
                int(small_bbox[2] * inv),
                int(small_bbox[3] * inv),
            )
        else:
            full_bbox = small_bbox

        # Clamp to image bounds.
        fx, fy, fw, fh = full_bbox
        fx = max(0, min(fx, orig_w - fw))
        fy = max(0, min(fy, orig_h - fh))
        fw = min(fw, orig_w - fx)
        fh = min(fh, orig_h - fy)
        full_bbox = (fx, fy, fw, fh)

        # Validate at full resolution with sub-cell offsets.
        # Try multiple offsets to find best grid alignment at full res,
        # matching the Phase 1 approach.
        p2_cell = fw // 16  # half-cell offset at full resolution
        best_p2_frac = 0.0
        best_p2_contrast = 0.0
        best_p2_bbox = full_bbox

        for p2_dy in range(-p2_cell, p2_cell + 1, max(1, p2_cell // 2)):
            for p2_dx in range(-p2_cell, p2_cell + 1, max(1, p2_cell // 2)):
                nx, ny = fx + p2_dx, fy + p2_dy
                if nx < 0 or ny < 0 or nx + fw > orig_w or ny + fh > orig_h:
                    continue
                region = gray_full[ny: ny + fh, nx: nx + fw]
                f, c = compute_alternation_strength(region)
                if f > best_p2_frac or (f == best_p2_frac and c > best_p2_contrast):
                    best_p2_frac = f
                    best_p2_contrast = c
                    best_p2_bbox = (nx, ny, fw, fh)

        if best_p2_frac < MIN_ALTERNATION_FRAC or best_p2_contrast < MIN_ALTERNATION_CONTRAST:
            continue

        # Validate with checkerboard consistency to eliminate false positives
        # from UI panels, banners, and other non-board regions.
        # Skip at low resolution — compression artifacts inflate checkerboard
        # std, causing false negatives on real boards.
        if min(orig_h, orig_w) >= 720:
            if not check_checkerboard_consistency(gray_full, best_p2_bbox):
                continue

        return OverlayDetection(
            found=True,
            bbox=best_p2_bbox,
            seed_bbox=best_p2_bbox,
            score=best_p2_frac,
            frame_resolution=resolution,
        )

    return OverlayDetection(found=False, frame_resolution=resolution)


# ---------------------------------------------------------------------------
# Fast overlay detection with accurate coordinates
# ---------------------------------------------------------------------------


def detect_overlay_fast(frame: np.ndarray) -> OverlayDetection:
    """Fast overlay detection with accurate coordinates.

    Uses ``fast_overlay_check`` as a seed, then expands the bbox via
    ``detect_grid`` to find pixel-perfect board boundaries.  Much faster
    than ``detect_overlay_in_frame`` because it skips the multi-scale
    sliding window expansion loop.

    Returns ``OverlayDetection`` with precise bbox when found, or
    ``found=False`` when no overlay is present.
    """
    from pipeline.overlay.grid_detector import (
        detect_grid,
        find_board_in_frame,
        grid_spacing_is_consistent,
    )

    h, w = frame.shape[:2]
    resolution = (w, h)
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    gray_full = cv2.GaussianBlur(gray_full, (3, 3), 0)

    # Phase 1: fast_overlay_check for seed detection.
    seed = fast_overlay_check(frame)
    if not seed.found or seed.bbox is None:
        grid_fallback = find_board_in_frame(frame)
        if (
            grid_fallback is not None
            and len(grid_fallback.v_lines) == 9
            and len(grid_fallback.h_lines) == 9
        ):
            fallback_bbox = _grid_result_to_bbox(grid_fallback, frame.shape)
            if _bbox_looks_like_overlay(frame, gray_full, fallback_bbox):
                return OverlayDetection(
                    found=True,
                    bbox=fallback_bbox,
                    score=compute_axis_aligned_periodicity(gray_full[
                        fallback_bbox[1] : fallback_bbox[1] + fallback_bbox[3],
                        fallback_bbox[0] : fallback_bbox[0] + fallback_bbox[2],
                    ]),
                    frame_resolution=resolution,
                )
        return OverlayDetection(found=False, frame_resolution=resolution)

    sx, sy, sw, sh = seed.bbox

    # Phase 2: expand outward from seed to find the full board.
    # The seed from fast_overlay_check is often a sub-region of the
    # actual board.  Try progressively larger windows centred on the
    # seed and keep the *largest* one where detect_grid succeeds.
    # Also try shifted centres (the seed may not be centred on the
    # true board) to handle off-centre detections.
    cx = sx + sw // 2
    cy = sy + sh // 2
    half = sw // 2
    centres = [
        (cx, cy),
        (cx - half, cy),
        (cx + half, cy),
        (cx, cy - half),
        (cx, cy + half),
    ]
    best: tuple[int, int, int, int] | None = None

    for ocx, ocy in centres:
        for mult_pct in range(125, 625, 25):
            size = int(sw * mult_pct / 100)
            if size > min(h, w):
                break
            ex = max(0, min(ocx - size // 2, w - size))
            ey = max(0, min(ocy - size // 2, h - size))
            size = min(size, w - ex, h - ey)
            exp_crop = frame[ey : ey + size, ex : ex + size]
            exp_blur = cv2.GaussianBlur(exp_crop, (3, 3), 0)
            grid = detect_grid(exp_blur, allow_uniform=False)
            if (
                grid is not None
                and len(grid.v_lines) == 9
                and len(grid.h_lines) == 9
            ):
                if best is None or size > best[2]:
                    best = (ex, ey, size, size)

    # Extract precise grid-line bbox from the best expansion.
    expansion_bbox: tuple[int, int, int, int] | None = None
    if best is not None:
        ex, ey, size, _ = best
        exp_crop = frame[ey : ey + size, ex : ex + size]
        exp_blur = cv2.GaussianBlur(exp_crop, (3, 3), 0)
        grid = detect_grid(exp_blur, allow_uniform=False)
        if (
            grid is not None
            and len(grid.v_lines) == 9
            and len(grid.h_lines) == 9
        ):
            gx = ex + grid.v_lines[0]
            gy = ey + grid.h_lines[0]
            gw = grid.v_lines[-1] - grid.v_lines[0]
            gh = grid.h_lines[-1] - grid.h_lines[0]
            board_size = max(gw, gh)
            gx = max(0, min(gx, w - board_size))
            gy = max(0, min(gy, h - board_size))
            board_size = min(board_size, w - gx, h - gy)
            expansion_bbox = (gx, gy, board_size, board_size)

    # Phase 2b: full-frame grid detection.
    # Large overlays (>40 % of frame) may have subtle internal grid lines
    # that Sobel misses on cropped regions but detects on the full frame
    # thanks to the strong board-to-background boundary edge.
    grid_full = find_board_in_frame(frame)
    if (
        grid_full is not None
        and len(grid_full.v_lines) == 9
        and len(grid_full.h_lines) == 9
        and grid_spacing_is_consistent(grid_full)
    ):
        grid_bbox = _grid_result_to_bbox(grid_full, frame.shape)
        gx1, gy1, _, _ = grid_bbox
        gx2, gy2 = grid_full.v_lines[-1], grid_full.h_lines[-1]
        seed_cx = sx + sw // 2
        seed_cy = sy + sh // 2
        if gx1 <= seed_cx <= gx2 and gy1 <= seed_cy <= gy2:
            full_board_size = grid_bbox[2]
            expansion_size = expansion_bbox[2] if expansion_bbox else 0
            if (
                full_board_size >= expansion_size
                and _bbox_looks_like_overlay(frame, gray_full, grid_bbox)
            ):
                gx, gy = grid_bbox[0], grid_bbox[1]
                return OverlayDetection(
                    found=True,
                    bbox=(gx, gy, full_board_size, full_board_size),
                    seed_bbox=seed.bbox,
                    score=max(seed.score, compute_axis_aligned_periodicity(
                        gray_full[gy : gy + full_board_size, gx : gx + full_board_size]
                    )),
                    frame_resolution=resolution,
                )

    # Return Phase 2 expansion result if it succeeded.
    if expansion_bbox is not None:
        return OverlayDetection(
            found=True,
            bbox=expansion_bbox,
            seed_bbox=seed.bbox,
            score=seed.score,
            frame_resolution=resolution,
            )

    # Phase 3: expansion didn't improve — use grid on the seed crop.
    crop = frame[
        max(0, sy) : min(h, sy + sh),
        max(0, sx) : min(w, sx + sw),
    ]
    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    grid = detect_grid(blurred)
    if grid is not None and len(grid.v_lines) == 9 and len(grid.h_lines) == 9:
        gx = max(0, sx) + grid.v_lines[0]
        gy = max(0, sy) + grid.h_lines[0]
        gw = grid.v_lines[-1] - grid.v_lines[0]
        gh = grid.h_lines[-1] - grid.h_lines[0]
        board_size = max(gw, gh)
        gx = max(0, min(gx, w - board_size))
        gy = max(0, min(gy, h - board_size))
        board_size = min(board_size, w - gx, h - gy)
        return OverlayDetection(
            found=True,
            bbox=(gx, gy, board_size, board_size),
            seed_bbox=seed.bbox,
            score=seed.score,
            frame_resolution=resolution,
        )

    # Phase 4: no grid found at all — return seed bbox as-is.
    return OverlayDetection(
        found=True,
        bbox=seed.bbox,
        seed_bbox=seed.bbox,
        score=seed.score,
        frame_resolution=resolution,
    )


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
        expanded_region = gray_check[by : by + bh, bx : bx + bw]
        seed_has_pattern = check_alternating_pattern(expanded_region)
        seed_area = best_seed_bbox[2] * best_seed_bbox[3]
        seed_fraction = best_seed_bbox[2] / min(h, w)
        if (
            not seed_has_pattern
            and seed_fraction < 0.50
            and best_expanded_area <= seed_area * 1.2
        ):
            return OverlayDetection(found=False, frame_resolution=resolution)

        # Expanded detections without the alternating pattern may still be
        # false positives.  Require consistent light/dark square colors.
        if not seed_has_pattern and not check_color_consistency(expanded_region):
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
