"""Strategy 1: Color segmentation + relative area ranking.

Segment piece pixels from auto-detected background, classify by brightness,
rank pieces by area within each color group to assign type.
"""

import numpy as np
from shared import (
    build_board,
    classify_empty,
    classify_piece_color,
    crop_squares,
    detect_orientation,
    detect_theme,
    find_board_in_frame,
    get_piece_mask,
)

STRATEGY_NAME = "S1: Area Ranking"


def read_position(frame: np.ndarray) -> str | None:
    """Read chess position from a video frame. Returns FEN or None."""
    grid = find_board_in_frame(frame)
    if grid is None:
        return None
    v_lines, h_lines, sq_size = grid
    squares = crop_squares(frame, v_lines, h_lines)
    light_bgr, dark_bgr = detect_theme(squares)
    empty_mask = classify_empty(squares)

    # Phase 1: classify empty/occupied and piece color
    piece_grid: list[list[str | None]] = [[None] * 8 for _ in range(8)]
    piece_areas: list[tuple[int, int, float, str]] = []  # (r, c, area_ratio, color)

    for r in range(8):
        for c in range(8):
            if empty_mask[r][c]:
                continue
            is_light_sq = (r + c) % 2 == 0
            bg_color = light_bgr if is_light_sq else dark_bgr

            color = classify_piece_color(squares[r][c], bg_color, light_bgr, dark_bgr)
            if color is None:
                continue

            mask = get_piece_mask(squares[r][c], bg_color)
            area_ratio = float(np.mean(mask))

            piece_areas.append((r, c, area_ratio, color))

    # Phase 2: assign piece types by relative area ranking within each color
    for color_name in ("white", "black"):
        color_pieces = [(r, c, area) for r, c, area, col in piece_areas if col == color_name]
        if not color_pieces:
            continue

        # Sort by area (ascending)
        color_pieces.sort(key=lambda x: x[2])

        n = len(color_pieces)

        # Assign types based on position in the sorted order.
        # Expected distribution: up to 8 pawns (smallest), then minor pieces,
        # then rooks, then queen/king (largest).
        # Use proportional assignment rather than fixed counts.
        for i, (r, c, area) in enumerate(color_pieces):
            frac = i / max(n - 1, 1)  # 0.0 = smallest, 1.0 = largest

            # Heuristic: assign by fractional position
            if frac < 0.55:
                piece_type = "p"  # pawn
            elif frac < 0.70:
                piece_type = "n"  # knight
            elif frac < 0.82:
                piece_type = "b"  # bishop
            elif frac < 0.92:
                piece_type = "r"  # rook
            else:
                piece_type = "q"  # queen

            # Position constraint: pawns can't be on display rows 0 or 7
            if piece_type == "p" and r in (0, 7):
                piece_type = "n"  # promote to minor piece

            sym = piece_type.upper() if color_name == "white" else piece_type
            piece_grid[r][c] = sym

    # Ensure we have kings — assign the largest unassigned piece per color as king
    for color_name in ("white", "black"):
        king_sym = "K" if color_name == "white" else "k"
        has_king = any(piece_grid[r][c] == king_sym for r in range(8) for c in range(8))
        if not has_king:
            # Find the largest queen of this color and make it king
            queen_sym = "Q" if color_name == "white" else "q"
            best_area = -1.0
            best_rc = None
            for r, c, area, col in piece_areas:
                if col == color_name and piece_grid[r][c] == queen_sym:
                    if area > best_area:
                        best_area = area
                        best_rc = (r, c)
            if best_rc:
                piece_grid[best_rc[0]][best_rc[1]] = king_sym
            else:
                # Fallback: make any piece of this color a king
                for r, c, area, col in piece_areas:
                    if col == color_name and piece_grid[r][c] is not None:
                        piece_grid[r][c] = king_sym
                        break

    # Detect orientation
    flipped = detect_orientation(piece_grid)
    board = build_board(piece_grid, flipped=flipped)
    return board.board_fen()
