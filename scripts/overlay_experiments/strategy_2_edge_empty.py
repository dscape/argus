"""Strategy 2: Edge-only empty detection + brightness-only classification.

Uses only edges for structure, only brightness for color.
Zero color/shape assumptions. Does NOT attempt piece type identification —
classifies as empty / white-piece / black-piece only.
This establishes a baseline for presence + color accuracy.
"""

import cv2
import numpy as np
from shared import (
    _inner_crop,
    build_board,
    crop_squares,
    detect_orientation,
    detect_theme,
    find_board_in_frame,
)

STRATEGY_NAME = "S2: Edge+Bright"


def read_position(frame: np.ndarray) -> str | None:
    """Read chess position from a video frame.

    Returns FEN. Since we don't identify piece types, all white pieces
    are encoded as pawns (P) and all black pieces as pawns (p), except
    we ensure one king per side by heuristic.
    """
    grid = find_board_in_frame(frame)
    if grid is None:
        return None
    v_lines, h_lines, sq_size = grid
    squares = crop_squares(frame, v_lines, h_lines)

    # Compute edge density per square
    edge_ratios = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            inner = _inner_crop(squares[r][c], 0.7)
            if inner.size == 0:
                continue
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
            # Auto Canny thresholds using Otsu
            high_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges = cv2.Canny(gray, high_thresh * 0.5, high_thresh)
            edge_ratios[r, c] = np.mean(edges > 0)

    # Adaptive threshold for empty detection using Otsu on the 64 edge ratios
    flat = edge_ratios.flatten()
    # Convert to uint8 range for Otsu
    flat_u8 = (flat * 255).astype(np.uint8)
    if np.max(flat_u8) > np.min(flat_u8):
        thresh_val, _ = cv2.threshold(flat_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        empty_threshold = thresh_val / 255.0
    else:
        empty_threshold = 0.02

    # Also detect theme for brightness comparison
    light_bgr, dark_bgr = detect_theme(squares)

    # Compute mean brightness of empty squares per square color
    empty_light_bright: list[float] = []
    empty_dark_bright: list[float] = []
    for r in range(8):
        for c in range(8):
            if edge_ratios[r, c] >= empty_threshold:
                continue  # occupied
            inner = _inner_crop(squares[r][c], 0.7)
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
            brightness = float(np.mean(gray))
            if (r + c) % 2 == 0:
                empty_light_bright.append(brightness)
            else:
                empty_dark_bright.append(brightness)

    # Fallback if no empty squares detected
    if not empty_light_bright:
        empty_light_bright = [float(np.mean(light_bgr))]
    if not empty_dark_bright:
        empty_dark_bright = [float(np.mean(dark_bgr))]

    ref_light = np.median(empty_light_bright)
    ref_dark = np.median(empty_dark_bright)

    # Classify each square
    piece_grid: list[list[str | None]] = [[None] * 8 for _ in range(8)]

    for r in range(8):
        for c in range(8):
            if edge_ratios[r, c] < empty_threshold:
                continue  # empty

            inner = _inner_crop(squares[r][c], 0.7)
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
            brightness = float(np.mean(gray))

            # Compare to empty square brightness for same board position
            ref = ref_light if (r + c) % 2 == 0 else ref_dark

            # Signed deviation — positive = brighter than expected, negative = darker
            deviation = brightness - ref

            # On light squares: white piece is bright (small deviation), black piece is dark (negative deviation)
            # On dark squares: white piece is bright (positive deviation), black piece is dark (small deviation)
            # Use the midpoint of theme colors as reference
            light_bright = float(
                np.mean(
                    cv2.cvtColor(light_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                )
            )
            dark_bright = float(
                np.mean(
                    cv2.cvtColor(dark_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                )
            )
            theme_midpoint = (light_bright + dark_bright) / 2.0

            if brightness > theme_midpoint:
                piece_grid[r][c] = "P"  # white piece (type unknown, use pawn)
            else:
                piece_grid[r][c] = "p"  # black piece

    # Ensure one king per side — pick the most central piece of each color
    for king_sym, pawn_sym in [("K", "P"), ("k", "p")]:
        has_king = any(piece_grid[r][c] == king_sym for r in range(8) for c in range(8))
        if not has_king:
            # Find the piece closest to expected king position
            best_dist = 999
            best_rc = None
            # Expected king column is e-file (col 4), back rank
            expected_r = 7 if pawn_sym == "P" else 0
            for r in range(8):
                for c in range(8):
                    if piece_grid[r][c] == pawn_sym:
                        dist = abs(r - expected_r) + abs(c - 4)
                        if dist < best_dist:
                            best_dist = dist
                            best_rc = (r, c)
            if best_rc:
                piece_grid[best_rc[0]][best_rc[1]] = king_sym

    flipped = detect_orientation(piece_grid)
    board = build_board(piece_grid, flipped=flipped)
    return board.board_fen()
