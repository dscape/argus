"""Robust chess board grid detection from video frames.

Two-stage approach:
1. Sobel edge projection — fast, works when the board dominates the region.
2. HoughLinesP fallback — handles partial views, noisy overlays, or small boards
   embedded in larger frames.

Both stages find 9 evenly-spaced vertical + 9 horizontal grid lines defining
the 8×8 board.
"""

import logging

import cv2
import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class GridResult:
    """Result of grid detection."""

    __slots__ = ("v_lines", "h_lines", "sq_size")

    def __init__(self, v_lines: list[int], h_lines: list[int], sq_size: int) -> None:
        self.v_lines = v_lines
        self.h_lines = h_lines
        self.sq_size = sq_size

    def crop_squares(self, image: np.ndarray) -> list[list[np.ndarray]]:
        """Crop the 64 squares. Returns squares[row][col]."""
        h, w = image.shape[:2]
        squares: list[list[np.ndarray]] = []
        for r in range(8):
            row: list[np.ndarray] = []
            for c in range(8):
                y1, y2 = self.h_lines[r], min(self.h_lines[r + 1], h)
                x1, x2 = self.v_lines[c], min(self.v_lines[c + 1], w)
                row.append(image[y1:y2, x1:x2])
            squares.append(row)
        return squares


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _find_consistent_subsequence(
    peaks: np.ndarray,
    min_spacing: int = 30,
    need: int = 9,
    tol: float = 0.10,
) -> tuple[list[int], int] | None:
    """Find the longest subsequence of *peaks* with consistent spacing.

    Returns ``(lines, spacing)`` with *lines* extended to *need* elements
    using uniform spacing, or ``None`` if fewer than ``need - 1`` consistent
    peaks are found.
    """
    if len(peaks) < need - 1:
        return None

    diffs = np.diff(peaks)
    best_lines: list[int] | None = None
    best_count = 0

    for d in diffs:
        if d < min_spacing:
            continue
        for start_idx in range(len(peaks)):
            lines = [int(peaks[start_idx])]
            for p in peaks[start_idx + 1 :]:
                expected = lines[-1] + d
                if abs(int(p) - expected) < d * tol:
                    lines.append(int(p))
            if len(lines) > best_count:
                best_count = len(lines)
                best_lines = list(lines)

    if best_lines is None or best_count < need - 1:
        return None

    # Compute median spacing from the ACTUAL matched peaks (not the
    # candidate diff) so the uniform grid stays close to every detected peak.
    actual_diffs = np.diff(best_lines)
    spacing = int(round(np.median(actual_diffs)))

    # Re-derive uniform lines anchored at the median start position.
    # Using a least-squares fit: find the start that minimises total
    # deviation from the detected peaks.
    n = len(best_lines)
    indices = np.arange(n)
    start = int(round(np.mean(np.array(best_lines) - indices * spacing)))
    uniform = [start + i * spacing for i in range(need)]

    return uniform, spacing


def _sobel_detect(gray: np.ndarray) -> GridResult | None:
    """Detect grid using Sobel edge projection."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    v_proj = np.sum(np.abs(sobel_x), axis=0)
    h_proj = np.sum(np.abs(sobel_y), axis=1)

    v_peaks, _ = find_peaks(v_proj, height=np.max(v_proj) * 0.25, distance=15)
    h_peaks, _ = find_peaks(h_proj, height=np.max(h_proj) * 0.25, distance=15)

    v_result = _find_consistent_subsequence(v_peaks)
    h_result = _find_consistent_subsequence(h_peaks)

    if v_result is None or h_result is None:
        return None

    v_lines, v_sp = v_result
    h_lines, h_sp = h_result

    # Spacings must be roughly equal (the board is square).
    if max(v_sp, h_sp) > min(v_sp, h_sp) * 1.3:
        return None

    sq_size = (v_sp + h_sp) // 2
    return GridResult(v_lines, h_lines, sq_size)


def _cluster_positions(positions: list[int], gap: int = 10) -> list[int]:
    """Merge nearby positions into cluster medians."""
    if not positions:
        return []
    positions = sorted(positions)
    clusters: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] < gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.median(c)) for c in clusters]


def _hough_detect(gray: np.ndarray) -> GridResult | None:
    """Detect grid using HoughLinesP on Canny edges."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10
    )
    if lines is None:
        return None

    h_positions: list[int] = []
    v_positions: list[int] = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1)) * 180 / np.pi
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 40:
            continue
        if angle < 15 or angle > 165:
            h_positions.append((y1 + y2) // 2)
        elif 75 < angle < 105:
            v_positions.append((x1 + x2) // 2)

    v_clustered = _cluster_positions(v_positions, gap=15)
    h_clustered = _cluster_positions(h_positions, gap=15)

    v_result = _find_consistent_subsequence(np.array(v_clustered), min_spacing=20)
    h_result = _find_consistent_subsequence(np.array(h_clustered), min_spacing=20)

    if v_result is None or h_result is None:
        return None

    v_lines, v_sp = v_result
    h_lines, h_sp = h_result

    if max(v_sp, h_sp) > min(v_sp, h_sp) * 1.3:
        return None

    sq_size = (v_sp + h_sp) // 2
    return GridResult(v_lines, h_lines, sq_size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_grid(image: np.ndarray) -> GridResult | None:
    """Detect the 8×8 grid in a (possibly cropped) image.

    Tries Sobel projection first, then falls back to HoughLinesP.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    result = _sobel_detect(gray)
    if result is not None:
        return result

    return _hough_detect(gray)


def find_board_in_frame(frame: np.ndarray) -> GridResult | None:
    """Find the chess board grid in a full video frame.

    Tries progressively smaller sub-regions until a valid grid is found.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # 1) Full frame
    result = detect_grid(frame)
    if result is not None:
        return result

    # 2) Sub-regions — left-heavy, right-heavy, center, halves
    regions = [
        (0, 0, w * 2 // 3, h),       # left 2/3
        (w // 3, 0, w, h),            # right 2/3
        (0, 0, w // 2, h),            # left half
        (w // 2, 0, w, h),            # right half
        (w // 4, 0, 3 * w // 4, h),   # centre half
        # Quadrants
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]

    for rx, ry, rx2, ry2 in regions:
        region = gray[ry:ry2, rx:rx2]
        if region.size == 0:
            continue
        result = detect_grid(cv2.cvtColor(region, cv2.COLOR_GRAY2BGR) if len(region.shape) == 2 else region)
        if result is None:
            # Try Hough directly on gray region
            result_h = _hough_detect(region)
            if result_h is not None:
                result = result_h
        if result is not None:
            # Offset to full-frame coordinates
            result.v_lines = [v + rx for v in result.v_lines]
            result.h_lines = [hv + ry for hv in result.h_lines]
            return result

    return None
