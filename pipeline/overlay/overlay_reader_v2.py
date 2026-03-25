"""Overlay reader v2 — self-bootstrapping template-based piece classification.

Approach A: Extract real piece templates from the starting position frame of
the video, then classify subsequent frames via NCC template matching.

Falls back to heuristic classification (color segmentation + area ranking)
when the starting position cannot be found.
"""

import logging

import chess
import cv2
import numpy as np

from pipeline.overlay.grid_detector import GridResult, find_board_in_frame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme & empty detection (auto-detected from the image)
# ---------------------------------------------------------------------------


def _inner_crop(cell: np.ndarray, frac: float = 0.6) -> np.ndarray:
    h, w = cell.shape[:2]
    m_y = max(1, int(h * (1 - frac) / 2))
    m_x = max(1, int(w * (1 - frac) / 2))
    return cell[m_y : h - m_y, m_x : w - m_x]


def _detect_theme(
    squares: list[list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Auto-detect light/dark board colors from lowest-variance squares."""
    variances: list[float] = []
    for r in range(8):
        for c in range(8):
            inner = _inner_crop(squares[r][c])
            if inner.size == 0:
                variances.append(1e9)
                continue
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            variances.append(float(np.var(gray)))

    sorted_idx = np.argsort(variances)
    empty_cands = sorted_idx[:24]

    light: list[np.ndarray] = []
    dark: list[np.ndarray] = []
    for idx in empty_cands:
        r, c = divmod(int(idx), 8)
        inner = _inner_crop(squares[r][c])
        if inner.size == 0:
            continue
        mean_col = np.mean(inner.reshape(-1, 3), axis=0)
        if (r + c) % 2 == 0:
            light.append(mean_col)
        else:
            dark.append(mean_col)

    if not light:
        light = dark
    if not dark:
        dark = light

    light_bgr = np.median(light, axis=0)
    dark_bgr = np.median(dark, axis=0)
    if np.sum(light_bgr) < np.sum(dark_bgr):
        light_bgr, dark_bgr = dark_bgr, light_bgr

    return light_bgr, dark_bgr


def _classify_empty(squares: list[list[np.ndarray]]) -> list[list[bool]]:
    """Variance gap analysis → empty/occupied mask."""
    variances = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            inner = _inner_crop(squares[r][c])
            if inner.size == 0:
                variances[r, c] = 1e9
                continue
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            variances[r, c] = float(np.var(gray))

    flat = variances.flatten()
    sorted_vals = np.sort(flat)

    max_gap = 0.0
    gap_idx = 0
    for i in range(5, 55):
        gap = sorted_vals[i + 1] - sorted_vals[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    threshold = (sorted_vals[gap_idx] + sorted_vals[gap_idx + 1]) / 2.0
    return [[bool(variances[r][c] < threshold) for c in range(8)] for r in range(8)]


# ---------------------------------------------------------------------------
# Piece color classification — dark-pixel fraction with gap-based threshold
# ---------------------------------------------------------------------------


def _classify_piece_colors(
    squares: list[list[np.ndarray]],
    empty: list[list[bool]],
    dark_bgr: np.ndarray,
) -> list[list[str | None]]:
    """Classify piece color for every occupied square.

    Black pieces have a solid dark body (high fraction of very-dark pixels).
    White pieces have thin dark outlines but a mostly bright body (low fraction).
    The threshold is derived from the gap in the distribution of dark-pixel
    fractions across all occupied squares.

    Returns grid of 'white', 'black', or None (for empty squares).
    """
    dark_gray = float(np.mean(
        cv2.cvtColor(dark_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ))
    # Pixels below this are "very dark" — clearly part of a piece body or outline
    dark_pixel_thresh = max(dark_gray - 60, 50)

    # Compute dark-pixel fraction for every occupied square
    fracs: dict[tuple[int, int], float] = {}
    for r in range(8):
        for c in range(8):
            if empty[r][c]:
                continue
            inner = _inner_crop(squares[r][c], 0.6)
            if inner.size == 0:
                continue
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
            fracs[(r, c)] = float(np.mean(gray < dark_pixel_thresh))

    if not fracs:
        return [[None] * 8 for _ in range(8)]

    # Use Otsu-style thresholding on the dark-frac values to find the
    # split between white (low dark-frac) and black (high dark-frac).
    # This is more robust than largest-gap because it maximises between-class
    # variance rather than finding the largest single gap.
    sorted_vals = sorted(fracs.values())
    n = len(sorted_vals)
    arr = np.array(sorted_vals)
    best_sigma = -1.0
    gap_split = 0.40  # fallback
    for i in range(1, n):
        w0 = i / n
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = np.mean(arr[:i])
        m1 = np.mean(arr[i:])
        sigma = w0 * w1 * (m0 - m1) ** 2
        if sigma > best_sigma:
            best_sigma = sigma
            gap_split = (arr[i - 1] + arr[i]) / 2.0

    result: list[list[str | None]] = [[None] * 8 for _ in range(8)]
    for (r, c), frac in fracs.items():
        result[r][c] = "black" if frac > gap_split else "white"
    return result


# ---------------------------------------------------------------------------
# NCC template matching
# ---------------------------------------------------------------------------

TEMPLATE_SIZE = 48


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation (scalar)."""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    return float(np.mean(a * b) / (sa * sb))


def _build_templates_from_starting_position(
    squares: list[list[np.ndarray]],
    flipped: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract per-piece templates from a starting position grid.

    Returns ``{piece_symbol: {"light": img, "dark": img}}`` where img is
    ``TEMPLATE_SIZE × TEMPLATE_SIZE`` grayscale.
    """
    if not flipped:
        rank8_row, rank7_row, rank2_row, rank1_row = 0, 1, 6, 7
    else:
        rank8_row, rank7_row, rank2_row, rank1_row = 7, 6, 1, 0

    file_order = list(range(8)) if not flipped else list(range(7, -1, -1))

    piece_map = {
        "r": (rank8_row, [file_order[0], file_order[7]]),
        "n": (rank8_row, [file_order[1], file_order[6]]),
        "b": (rank8_row, [file_order[2], file_order[5]]),
        "q": (rank8_row, [file_order[3]]),
        "k": (rank8_row, [file_order[4]]),
        "p": (rank7_row, file_order),
        "R": (rank1_row, [file_order[0], file_order[7]]),
        "N": (rank1_row, [file_order[1], file_order[6]]),
        "B": (rank1_row, [file_order[2], file_order[5]]),
        "Q": (rank1_row, [file_order[3]]),
        "K": (rank1_row, [file_order[4]]),
        "P": (rank2_row, file_order),
    }

    templates: dict[str, dict[str, np.ndarray]] = {}
    for sym, (row, cols) in piece_map.items():
        light_crops: list[np.ndarray] = []
        dark_crops: list[np.ndarray] = []
        for col in cols:
            cell = squares[row][col]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
            resized = cv2.resize(gray, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            is_light = (row + col) % 2 == 0
            (light_crops if is_light else dark_crops).append(resized)

        tpl: dict[str, np.ndarray] = {}
        if light_crops:
            tpl["light"] = np.mean(light_crops, axis=0).astype(np.uint8)
        if dark_crops:
            tpl["dark"] = np.mean(dark_crops, axis=0).astype(np.uint8)
        if "light" not in tpl and "dark" in tpl:
            tpl["light"] = tpl["dark"]
        if "dark" not in tpl and "light" in tpl:
            tpl["dark"] = tpl["light"]
        templates[sym] = tpl

    return templates


def _classify_with_templates(
    square: np.ndarray,
    is_light: bool,
    piece_color: str,
    templates: dict[str, dict[str, np.ndarray]],
) -> str | None:
    """NCC-match a square against templates; return piece symbol."""
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if len(square.shape) == 3 else square
    resized = cv2.resize(gray, (TEMPLATE_SIZE, TEMPLATE_SIZE))
    sq_key = "light" if is_light else "dark"

    if piece_color == "white":
        candidates = [s for s in templates if s.isupper()]
    else:
        candidates = [s for s in templates if s.islower()]

    best_sym: str | None = None
    best_score = -1.0
    for sym in candidates:
        tpl = templates[sym].get(sq_key)
        if tpl is None:
            continue
        score = _ncc(resized, tpl)
        if score > best_score:
            best_score = score
            best_sym = sym

    return best_sym


# ---------------------------------------------------------------------------
# Starting position finder
# ---------------------------------------------------------------------------


def _find_starting_position_frame(
    video_path: str,
    max_seconds: int = 120,
) -> tuple[np.ndarray, GridResult, bool] | None:
    """Scan the first *max_seconds* for a frame showing the starting position.

    Returns (frame, grid, flipped) or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    for t_sec in range(0, max_seconds, 2):
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        grid = find_board_in_frame(frame)
        if grid is None:
            continue

        squares = grid.crop_squares(frame)
        empty = _classify_empty(squares)
        n_occupied = sum(1 for r in range(8) for c in range(8) if not empty[r][c])

        # Starting position has exactly 32 pieces
        if n_occupied < 28 or n_occupied > 36:
            continue

        # Empty squares should be in middle ranks (rows 2-5 in image)
        middle_empty = sum(
            1 for r in range(2, 6) for c in range(8) if empty[r][c]
        )
        if middle_empty < 24:
            continue

        top_occupied = sum(1 for c in range(8) if not empty[0][c])
        bot_occupied = sum(1 for c in range(8) if not empty[7][c])

        if top_occupied >= 6 and bot_occupied >= 6:
            _, dark_bgr = _detect_theme(squares)
            colors = _classify_piece_colors(squares, empty, dark_bgr)

            # Count white pieces in top vs bottom row
            top_white = sum(1 for c in range(8) if colors[0][c] == "white")
            bot_white = sum(1 for c in range(8) if colors[7][c] == "white")

            # In standard orientation, row 0 = rank 8 = black, row 7 = rank 1 = white
            flipped = top_white > bot_white
            cap.release()
            return frame, grid, flipped

    cap.release()
    return None


# ---------------------------------------------------------------------------
# Orientation detection
# ---------------------------------------------------------------------------


def _detect_orientation(piece_grid: list[list[str | None]]) -> bool:
    """Return True if the board is flipped (black on bottom)."""

    def _score(flipped: bool) -> float:
        board = chess.Board(fen=None)
        for r in range(8):
            for c in range(8):
                sym = piece_grid[r][c]
                if sym is None:
                    continue
                piece = chess.Piece.from_symbol(sym)
                if not flipped:
                    sq = chess.square(c, 7 - r)
                else:
                    sq = chess.square(7 - c, r)
                board.set_piece_at(sq, piece)

        score = 0.0
        wk = len(board.pieces(chess.KING, chess.WHITE))
        bk = len(board.pieces(chess.KING, chess.BLACK))
        if wk == 1:
            score += 10
        if bk == 1:
            score += 10
        for sq in chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_8):
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN:
                score -= 5
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p is None:
                continue
            rank = chess.square_rank(sq)
            if p.color == chess.WHITE and p.piece_type != chess.PAWN and rank <= 1:
                score += 1
            if p.color == chess.BLACK and p.piece_type != chess.PAWN and rank >= 6:
                score += 1
        return score

    return _score(True) > _score(False)


# ---------------------------------------------------------------------------
# Build FEN
# ---------------------------------------------------------------------------


def _build_fen(piece_grid: list[list[str | None]], flipped: bool) -> str:
    board = chess.Board(fen=None)
    for r in range(8):
        for c in range(8):
            sym = piece_grid[r][c]
            if sym is None:
                continue
            piece = chess.Piece.from_symbol(sym)
            if not flipped:
                sq = chess.square(c, 7 - r)
            else:
                sq = chess.square(7 - c, r)
            board.set_piece_at(sq, piece)
    return board.board_fen()


# ---------------------------------------------------------------------------
# Heuristic fallback (when no starting position is found)
# ---------------------------------------------------------------------------


def _heuristic_piece_type(
    square: np.ndarray,
    bg_color: np.ndarray,
    piece_color: str,
    all_occupied: list[tuple[int, int, float, str]],
) -> str:
    """Rough piece-type guess from area ranking within the same color group."""
    inner = _inner_crop(square, 0.7)
    diff = np.linalg.norm(inner.astype(float) - bg_color.astype(float), axis=2)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    piece_mask = diff > max(float(thresh_val), 20.0)
    area = float(np.mean(piece_mask))

    same_color = sorted(
        [(r, c, a, col) for r, c, a, col in all_occupied if col == piece_color],
        key=lambda x: x[2],
    )
    n = len(same_color)
    rank = next(
        (i for i, (_, _, a, _) in enumerate(same_color) if abs(a - area) < 0.001),
        n // 2,
    )

    if n <= 2:
        return "K" if rank == n - 1 else "Q"
    frac = rank / n
    if frac < 0.55:
        return "P"
    elif frac < 0.70:
        return "N"
    elif frac < 0.80:
        return "B"
    elif frac < 0.90:
        return "R"
    else:
        return "Q"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_fen_from_frame(
    frame: np.ndarray,
    video_path: str | None = None,
) -> str | None:
    """Read the chess position from a video frame.

    If *video_path* is given, attempts to bootstrap templates from the
    starting position in the video for accurate piece-type identification.

    Returns piece-placement FEN or None.
    """
    grid = find_board_in_frame(frame)
    if grid is None:
        logger.warning("No grid found in frame")
        return None

    squares = grid.crop_squares(frame)
    empty = _classify_empty(squares)
    light_bgr, dark_bgr = _detect_theme(squares)

    # Piece color classification (gap-based)
    colors = _classify_piece_colors(squares, empty, dark_bgr)

    # Try to get templates from the video's starting position
    templates: dict[str, dict[str, np.ndarray]] | None = None
    bootstrap_flipped: bool | None = None
    if video_path is not None:
        bootstrap = _find_starting_position_frame(video_path)
        if bootstrap is not None:
            bp_frame, bp_grid, bp_flipped = bootstrap
            bp_squares = bp_grid.crop_squares(bp_frame)
            templates = _build_templates_from_starting_position(bp_squares, bp_flipped)
            bootstrap_flipped = bp_flipped

    # Classify each occupied square
    piece_grid: list[list[str | None]] = [[None] * 8 for _ in range(8)]
    all_occupied: list[tuple[int, int, float, str]] = []

    for r in range(8):
        for c in range(8):
            pc = colors[r][c]
            if pc is None:
                continue  # empty

            is_light = (r + c) % 2 == 0

            # NCC template matching (if bootstrap succeeded)
            if templates is not None:
                sym = _classify_with_templates(squares[r][c], is_light, pc, templates)
                if sym is not None:
                    piece_grid[r][c] = sym
                    continue

            # Compute area for heuristic fallback
            bg = light_bgr if is_light else dark_bgr
            inner = _inner_crop(squares[r][c], 0.7)
            diff = np.linalg.norm(inner.astype(float) - bg.astype(float), axis=2)
            diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
            tv, _ = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = diff > max(float(tv), 20.0)
            area = float(np.mean(mask))
            all_occupied.append((r, c, area, pc))

    # Heuristic fallback for squares without template match
    for r, c, area, pc in all_occupied:
        is_light = (r + c) % 2 == 0
        bg = light_bgr if is_light else dark_bgr
        ptype = _heuristic_piece_type(squares[r][c], bg, pc, all_occupied)
        piece_grid[r][c] = ptype if pc == "white" else ptype.lower()

    # Detect orientation
    if bootstrap_flipped is not None:
        flipped = bootstrap_flipped
    else:
        flipped = _detect_orientation(piece_grid)

    return _build_fen(piece_grid, flipped)
