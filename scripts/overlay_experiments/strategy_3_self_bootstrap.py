"""Strategy 3: Self-bootstrapping from known position.

Find the starting position in the video, extract real piece templates,
use NCC matching against them for all subsequent frames.
"""

import cv2
import numpy as np
from shared import (
    _inner_crop,
    build_board,
    classify_empty,
    crop_squares,
    detect_theme,
    extract_frame,
    find_board_in_frame,
)

STRATEGY_NAME = "S3: Self-Boot"

# Starting position piece layout (display row 0 = rank 8)
STARTING_LAYOUT = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],  # rank 8
    ["p", "p", "p", "p", "p", "p", "p", "p"],  # rank 7
    [None] * 8,  # rank 6
    [None] * 8,  # rank 5
    [None] * 8,  # rank 4
    [None] * 8,  # rank 3
    ["P", "P", "P", "P", "P", "P", "P", "P"],  # rank 2
    ["R", "N", "B", "Q", "K", "B", "N", "R"],  # rank 1
]

# Also support flipped starting position
STARTING_LAYOUT_FLIPPED = [
    ["R", "N", "B", "K", "Q", "B", "N", "R"],  # rank 1 (flipped: at top)
    ["P", "P", "P", "P", "P", "P", "P", "P"],  # rank 2
    [None] * 8,
    [None] * 8,
    [None] * 8,
    [None] * 8,
    ["p", "p", "p", "p", "p", "p", "p", "p"],  # rank 7
    ["r", "n", "b", "k", "q", "b", "n", "r"],  # rank 8
]


def _is_starting_position(empty_mask: list[list[bool]]) -> tuple[bool, bool]:
    """Check if empty_mask matches starting position pattern.

    Returns (is_start, is_flipped).
    Starting position: rows 2-5 empty, rows 0-1 and 6-7 occupied.
    """
    n_empty = sum(empty_mask[r][c] for r in range(8) for c in range(8))
    if n_empty < 28 or n_empty > 36:
        return False, False

    # Normal orientation: rows 2-5 should be mostly empty, 0-1 and 6-7 occupied
    empty_middle = sum(empty_mask[r][c] for r in range(2, 6) for c in range(8))
    occ_edges = sum(not empty_mask[r][c] for r in [0, 1, 6, 7] for c in range(8))

    if empty_middle >= 28 and occ_edges >= 28:
        return True, False

    # Flipped: same pattern (symmetric)
    return True, True  # Can't distinguish from pattern alone


def _build_templates(
    squares: list[list[np.ndarray]],
    layout: list[list[str | None]],
    light_bgr: np.ndarray,
    dark_bgr: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    """Build template library from known position.

    Returns dict mapping piece key (e.g., "P_light", "n_dark") to list of
    template images (inner-cropped, resized to canonical size).
    """
    CANONICAL = 32
    templates: dict[str, list[np.ndarray]] = {}

    for r in range(8):
        for c in range(8):
            piece_sym = layout[r][c]
            if piece_sym is None:
                continue
            sq_type = "light" if (r + c) % 2 == 0 else "dark"
            key = f"{piece_sym}_{sq_type}"
            inner = _inner_crop(squares[r][c], 0.7)
            if inner.size == 0:
                continue
            resized = cv2.resize(inner, (CANONICAL, CANONICAL))
            templates.setdefault(key, []).append(resized)

    # Average multiple instances
    averaged: dict[str, list[np.ndarray]] = {}
    for key, imgs in templates.items():
        avg = np.mean([img.astype(float) for img in imgs], axis=0).astype(np.uint8)
        averaged[key] = [avg]
        # Also keep individual templates for diversity
        if len(imgs) > 1:
            averaged[key].extend(imgs[:3])

    return averaged


def _ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized cross-correlation between two images."""
    g1 = (
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(float)
        if len(img1.shape) == 3
        else img1.astype(float)
    )
    g2 = (
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(float)
        if len(img2.shape) == 3
        else img2.astype(float)
    )

    g1 = g1 - np.mean(g1)
    g2 = g2 - np.mean(g2)

    std1 = np.std(g1)
    std2 = np.std(g2)
    if std1 < 1e-6 or std2 < 1e-6:
        return 0.0

    return float(np.mean(g1 * g2) / (std1 * std2))


def _classify_with_templates(
    square: np.ndarray,
    sq_type: str,
    templates: dict[str, list[np.ndarray]],
) -> str | None:
    """Classify a square using NCC against templates.

    Returns piece symbol or None if best match is empty.
    """
    CANONICAL = 32
    inner = _inner_crop(square, 0.7)
    if inner.size == 0:
        return None
    resized = cv2.resize(inner, (CANONICAL, CANONICAL))

    best_score = -1.0
    best_piece = None

    for key, tmpls in templates.items():
        piece_sym, tmpl_sq_type = key.rsplit("_", 1)
        if tmpl_sq_type != sq_type:
            continue

        for tmpl in tmpls:
            score = _ncc(resized, tmpl)
            if score > best_score:
                best_score = score
                best_piece = piece_sym

    # Only accept if NCC is reasonably high
    if best_score < 0.3:
        return None

    return best_piece


class SelfBootstrapReader:
    """Reader that bootstraps templates from the starting position."""

    def __init__(self):
        self.templates: dict[str, list[np.ndarray]] | None = None
        self.flipped: bool = False

    def bootstrap(self, video_path: str) -> bool:
        """Scan the video for starting position and extract templates.

        Returns True if bootstrap succeeded.
        """
        for t in range(2, 60, 2):
            frame = extract_frame(video_path, t)
            if frame is None:
                continue

            grid = find_board_in_frame(frame)
            if grid is None:
                continue

            v_lines, h_lines, sq_size = grid
            squares = crop_squares(frame, v_lines, h_lines)
            empty_mask = classify_empty(squares)

            is_start, is_flipped = _is_starting_position(empty_mask)
            if not is_start:
                continue

            light_bgr, dark_bgr = detect_theme(squares)
            layout = STARTING_LAYOUT_FLIPPED if is_flipped else STARTING_LAYOUT
            self.templates = _build_templates(squares, layout, light_bgr, dark_bgr)
            self.flipped = is_flipped
            return True

        return False

    def read_position(self, frame: np.ndarray) -> str | None:
        """Read position using bootstrapped templates."""
        if self.templates is None:
            return None

        grid = find_board_in_frame(frame)
        if grid is None:
            return None

        v_lines, h_lines, sq_size = grid
        squares = crop_squares(frame, v_lines, h_lines)
        empty_mask = classify_empty(squares)

        piece_grid: list[list[str | None]] = [[None] * 8 for _ in range(8)]

        for r in range(8):
            for c in range(8):
                if empty_mask[r][c]:
                    continue
                sq_type = "light" if (r + c) % 2 == 0 else "dark"
                piece = _classify_with_templates(squares[r][c], sq_type, self.templates)
                piece_grid[r][c] = piece

        # Ensure kings
        for king_sym in ("K", "k"):
            has_king = any(piece_grid[r][c] == king_sym for r in range(8) for c in range(8))
            if not has_king:
                # Find the best candidate by NCC with king template
                sq_candidates = []
                for r in range(8):
                    for c in range(8):
                        if piece_grid[r][c] is not None and piece_grid[r][c] != king_sym:
                            p = piece_grid[r][c]
                            if (king_sym == "K" and p.isupper()) or (
                                king_sym == "k" and p.islower()
                            ):
                                sq_candidates.append((r, c))
                if sq_candidates:
                    # Pick the one on the most typical king rank
                    back_rank = 7 if king_sym == "K" else 0
                    sq_candidates.sort(key=lambda rc: abs(rc[0] - back_rank))
                    r, c = sq_candidates[0]
                    piece_grid[r][c] = king_sym

        board = build_board(piece_grid, flipped=self.flipped)
        return board.board_fen()


# Module-level function for the harness
_reader: SelfBootstrapReader | None = None


def read_position(frame: np.ndarray, video_path: str | None = None) -> str | None:
    """Read position from frame. Pass video_path on first call to bootstrap."""
    global _reader
    if _reader is None or video_path is not None:
        _reader = SelfBootstrapReader()
        if video_path:
            if not _reader.bootstrap(video_path):
                # Fallback: try strategy 1
                from strategy_1_area_ranking import read_position as fallback

                return fallback(frame)
    return _reader.read_position(frame)
