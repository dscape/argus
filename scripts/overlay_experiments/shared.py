"""Shared utilities for overlay reader experiments.

Provides grid detection, theme detection, empty/occupied classification,
orientation detection, and evaluation — all auto-detected from the image
with zero hardcoded assumptions about theme, piece style, or orientation.
"""

import sys
from pathlib import Path

import chess
import cv2
import numpy as np
from scipy.signal import find_peaks

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Video paths
# ---------------------------------------------------------------------------
VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "STLChessClub"
VIDEO_IDS = ["O8ZwstOxG_A", "7RaBQag34Hk", "2wWUKmCBr6A", "Ov8PXnJp1PU"]

# ---------------------------------------------------------------------------
# Ground truth FENs  (piece-placement only, manually read from frames)
# Key: (video_id_prefix, timestamp_seconds)
# ---------------------------------------------------------------------------
GROUND_TRUTH: dict[tuple[str, int], str] = {
    # O8ZwstOxG_A @ 60s — verified by user (b1 is empty)
    ("O8Z", 60): "r3rn1k/p1pq1ppp/bp3n2/3p1N2/3P4/PPB3PB/4PP1P/R2QR1K1 w - - 0 1",
    # 7RaBQag34Hk @ 60s — verified by user
    ("7Ra", 60): "r1bqkb1r/pp1n1pp1/2n1p2p/2ppP3/3P3P/2P2NP1/PP3P2/RNBQKB1R w - - 0 1",
    # 2wWUKmCBr6A @ 60s — verified by user
    ("2wW", 60): "r1b2rk1/pp3ppp/2n1pn2/2b5/2P5/5NP1/P2NPPBP/R1BR2K1 w - - 0 1",
    # Ov8PXnJp1PU @ 60s — endgame, verified by user
    ("Ov8", 60): "8/8/8/1p6/pBb4p/P4k2/7P/6K1 w - - 0 1",
}

# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------


def _find_consistent_peaks(
    peaks: np.ndarray, min_spacing: int = 30
) -> tuple[list[int], int] | None:
    """Find the longest subsequence of peaks with consistent spacing.

    Returns (lines, spacing) or None if fewer than 8 consistent peaks found.
    """
    if len(peaks) < 8:
        return None
    diffs = np.diff(peaks)
    best_lines: list[int] | None = None
    best_count = 0
    best_spacing = 0

    for d in diffs:
        if d < min_spacing:
            continue
        for start_idx in range(len(peaks)):
            lines = [int(peaks[start_idx])]
            for p in peaks[start_idx + 1 :]:
                expected = lines[-1] + d
                if abs(int(p) - expected) < d * 0.10:  # tighter tolerance
                    lines.append(int(p))
            if len(lines) > best_count:
                best_count = len(lines)
                best_lines = lines
                best_spacing = int(d)

    if best_lines is None or best_count < 7:
        return None

    # Re-derive lines using uniform spacing from the best start point.
    # This fixes drift when detected peaks are slightly off.
    start = best_lines[0]
    uniform_lines = [start + i * best_spacing for i in range(9)]

    return uniform_lines, best_spacing


def detect_grid(image: np.ndarray) -> tuple[list[int], list[int], int] | None:
    """Detect the 8x8 chess board grid in an image using Sobel edge projection.

    Returns (v_lines, h_lines, square_size) where v_lines and h_lines each
    contain 9 pixel coordinates (the boundaries of the 8 columns/rows).
    Returns None if no valid grid is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    vert_proj = np.sum(np.abs(sobel_x), axis=0)
    horiz_proj = np.sum(np.abs(sobel_y), axis=1)

    # Adaptive peak height threshold
    v_peaks, _ = find_peaks(vert_proj, height=np.max(vert_proj) * 0.25, distance=15)
    h_peaks, _ = find_peaks(horiz_proj, height=np.max(horiz_proj) * 0.25, distance=15)

    v_result = _find_consistent_peaks(v_peaks)
    h_result = _find_consistent_peaks(h_peaks)

    if v_result is None or h_result is None:
        return None

    v_lines, v_spacing = v_result
    h_lines, h_spacing = h_result

    # Spacings should be roughly equal (square board)
    if max(v_spacing, h_spacing) > min(v_spacing, h_spacing) * 1.3:
        return None

    sq_size = (v_spacing + h_spacing) // 2
    return v_lines, h_lines, sq_size


def crop_squares(
    image: np.ndarray, v_lines: list[int], h_lines: list[int]
) -> list[list[np.ndarray]]:
    """Crop the 64 squares from the image. Returns squares[row][col]."""
    squares: list[list[np.ndarray]] = []
    for r in range(8):
        row: list[np.ndarray] = []
        for c in range(8):
            y1, y2 = h_lines[r], h_lines[r + 1]
            x1, x2 = v_lines[c], min(v_lines[c + 1], image.shape[1])
            y2 = min(y2, image.shape[0])
            cell = image[y1:y2, x1:x2]
            row.append(cell)
        squares.append(row)
    return squares


def _inner_crop(cell: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """Return the inner portion of a cell (avoids grid-line artifacts)."""
    h, w = cell.shape[:2]
    margin_y = int(h * (1 - frac) / 2)
    margin_x = int(w * (1 - frac) / 2)
    if margin_y < 1:
        margin_y = 1
    if margin_x < 1:
        margin_x = 1
    return cell[margin_y : h - margin_y, margin_x : w - margin_x]


# ---------------------------------------------------------------------------
# Theme detection
# ---------------------------------------------------------------------------


def detect_theme(squares: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Auto-detect board theme colors from the squares.

    Returns (light_bgr, dark_bgr) as float arrays.
    """
    variances = []
    for r in range(8):
        for c in range(8):
            inner = _inner_crop(squares[r][c])
            if inner.size == 0:
                variances.append(1e9)
                continue
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
            variances.append(float(np.var(gray)))

    # Lowest-variance 24 squares are likely empty
    sorted_idx = np.argsort(variances)
    empty_candidates = sorted_idx[:24]

    light_colors: list[np.ndarray] = []
    dark_colors: list[np.ndarray] = []
    for idx in empty_candidates:
        r, c = divmod(int(idx), 8)
        inner = _inner_crop(squares[r][c])
        if inner.size == 0:
            continue
        mean_col = np.mean(inner.reshape(-1, 3), axis=0)
        if (r + c) % 2 == 0:
            light_colors.append(mean_col)
        else:
            dark_colors.append(mean_col)

    # Fallback if one group is empty
    if not light_colors:
        light_colors = dark_colors
    if not dark_colors:
        dark_colors = light_colors

    light_bgr = np.median(light_colors, axis=0)
    dark_bgr = np.median(dark_colors, axis=0)

    # Ensure light is actually lighter
    if np.sum(light_bgr) < np.sum(dark_bgr):
        light_bgr, dark_bgr = dark_bgr, light_bgr

    return light_bgr, dark_bgr


# ---------------------------------------------------------------------------
# Empty / occupied classification
# ---------------------------------------------------------------------------


def classify_empty(squares: list[list[np.ndarray]]) -> list[list[bool]]:
    """Classify each square as empty (True) or occupied (False).

    Uses variance gap analysis — no fixed threshold.
    """
    variances = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            inner = _inner_crop(squares[r][c])
            if inner.size == 0:
                variances[r, c] = 1e9
                continue
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
            variances[r, c] = float(np.var(gray))

    flat = variances.flatten()
    sorted_vals = np.sort(flat)

    # Find the largest gap between consecutive sorted variances.
    # The gap separates the "empty" cluster from the "occupied" cluster.
    max_gap = 0.0
    gap_idx = 0
    for i in range(5, 55):
        gap = sorted_vals[i + 1] - sorted_vals[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    threshold = (sorted_vals[gap_idx] + sorted_vals[gap_idx + 1]) / 2.0

    result: list[list[bool]] = []
    for r in range(8):
        row: list[bool] = []
        for c in range(8):
            row.append(bool(variances[r, c] < threshold))
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Piece color classification
# ---------------------------------------------------------------------------


def classify_piece_color(
    square: np.ndarray,
    bg_color: np.ndarray,
    light_bgr: np.ndarray,
    dark_bgr: np.ndarray,
) -> str | None:
    """Classify a piece as 'white' or 'black' based on its pixel brightness.

    Args:
        square: BGR image of the square.
        bg_color: Expected background color for this square.
        light_bgr, dark_bgr: Auto-detected theme colors.

    Returns 'white', 'black', or None if unclear.
    """
    inner = _inner_crop(square, 0.7)
    if inner.size == 0:
        return None

    # Per-pixel color distance from expected background
    diff = np.linalg.norm(inner.astype(float) - bg_color.astype(float), axis=2)

    # Adaptive threshold using Otsu on the distance image
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    piece_mask = diff > max(thresh_val, 20)

    piece_ratio = np.mean(piece_mask)
    if piece_ratio < 0.03:
        return None  # Probably empty

    # Mean brightness of piece pixels
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
    piece_brightness = float(np.mean(gray[piece_mask]))

    # Compare to theme midpoint
    light_bright = float(
        np.mean(cv2.cvtColor(light_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY))
    )
    dark_bright = float(
        np.mean(cv2.cvtColor(dark_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY))
    )
    midpoint = (light_bright + dark_bright) / 2.0

    return "white" if piece_brightness > midpoint else "black"


def get_piece_mask(
    square: np.ndarray,
    bg_color: np.ndarray,
) -> np.ndarray:
    """Get a binary mask of piece pixels in a square.

    Returns a boolean mask the same size as the inner crop.
    """
    inner = _inner_crop(square, 0.7)
    if inner.size == 0:
        return np.zeros((1, 1), dtype=bool)

    diff = np.linalg.norm(inner.astype(float) - bg_color.astype(float), axis=2)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return diff > max(thresh_val, 20)


# ---------------------------------------------------------------------------
# Orientation detection
# ---------------------------------------------------------------------------


def detect_orientation(piece_grid: list[list[str | None]]) -> bool:
    """Determine if the board is flipped (black on bottom).

    piece_grid[r][c] = piece symbol ('P','N',... 'p','n',...) or None.

    Returns True if flipped (black on bottom), False if normal (white on bottom).
    """

    def _score_orientation(grid: list[list[str | None]], flipped: bool) -> float:
        """Score how chess-legal a piece arrangement looks."""
        board = chess.Board(fen=None)
        for r in range(8):
            for c in range(8):
                piece_sym = grid[r][c]
                if piece_sym is None:
                    continue
                piece = chess.Piece.from_symbol(piece_sym)
                if not flipped:
                    sq = chess.square(c, 7 - r)
                else:
                    sq = chess.square(7 - c, r)
                board.set_piece_at(sq, piece)

        score = 0.0

        # Must have exactly 1 king per side
        wk = len(board.pieces(chess.KING, chess.WHITE))
        bk = len(board.pieces(chess.KING, chess.BLACK))
        if wk == 1:
            score += 10
        if bk == 1:
            score += 10

        # No pawns on ranks 1 or 8
        for sq in chess.SquareSet(chess.BB_RANK_1 | chess.BB_RANK_8):
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN:
                score -= 5

        # White heavy pieces should be on lower ranks, black on upper
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p is None:
                continue
            rank = chess.square_rank(sq)
            if p.color == chess.WHITE and p.piece_type != chess.PAWN:
                if rank <= 1:
                    score += 1
            elif p.color == chess.BLACK and p.piece_type != chess.PAWN:
                if rank >= 6:
                    score += 1

        return score

    score_normal = _score_orientation(piece_grid, flipped=False)
    score_flipped = _score_orientation(piece_grid, flipped=True)
    return score_flipped > score_normal


# ---------------------------------------------------------------------------
# Board building
# ---------------------------------------------------------------------------


def build_board(
    piece_grid: list[list[str | None]],
    flipped: bool = False,
) -> chess.Board:
    """Build a chess.Board from a piece grid.

    piece_grid[r][c] = piece symbol or None.
    r=0 is the top row of the image.
    """
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
    return board


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frame(video_path: str | Path, timestamp_sec: float) -> np.ndarray | None:
    """Extract a single frame from a video at the given timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Board-in-frame finder
# ---------------------------------------------------------------------------


def find_board_in_frame(frame: np.ndarray) -> tuple[list[int], list[int], int] | None:
    """Find the chess board grid in a full video frame.

    Tries the full frame first, then scans overlapping regions.
    """
    h, w = frame.shape[:2]

    # Try full frame
    result = detect_grid(frame)
    if result is not None:
        return result

    # Try left half, right half, and quadrants
    regions = [
        (0, 0, w * 2 // 3, h),  # left 2/3
        (w // 3, 0, w, h),  # right 2/3
        (0, 0, w // 2, h),  # left half
        (w // 2, 0, w, h),  # right half
        (w // 4, 0, 3 * w // 4, h),  # center half
    ]

    for rx, ry, rx2, ry2 in regions:
        region = frame[ry:ry2, rx:rx2]
        result = detect_grid(region)
        if result is not None:
            v_lines, h_lines, sq_size = result
            # Offset to full-frame coordinates
            v_lines = [v + rx for v in v_lines]
            h_lines = [h_val + ry for h_val in h_lines]
            return v_lines, h_lines, sq_size

    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(predicted_fen: str | None, ground_truth_fen: str) -> dict:
    """Compare predicted FEN against ground truth.

    Both should be full FEN strings (only piece placement is compared).
    Returns dict with accuracy metrics.
    """
    if predicted_fen is None:
        return {
            "empty_acc": 0.0,
            "presence_acc": 0.0,
            "color_acc": 0.0,
            "type_acc": 0.0,
            "exact_match": False,
        }

    pred_board = chess.Board(
        predicted_fen if " " in predicted_fen else predicted_fen + " w - - 0 1"
    )
    gt_board = chess.Board(
        ground_truth_fen if " " in ground_truth_fen else ground_truth_fen + " w - - 0 1"
    )

    empty_correct = 0
    presence_correct = 0
    color_correct = 0
    type_correct = 0
    total_occupied_gt = 0

    for sq in chess.SQUARES:
        pred_piece = pred_board.piece_at(sq)
        gt_piece = gt_board.piece_at(sq)

        pred_empty = pred_piece is None
        gt_empty = gt_piece is None

        # Empty accuracy
        if pred_empty == gt_empty:
            empty_correct += 1
            presence_correct += 1
        # Presence (correct if both empty or both occupied)
        # already counted above

        if not gt_empty:
            total_occupied_gt += 1
            if not pred_empty:
                # Color accuracy
                if pred_piece.color == gt_piece.color:
                    color_correct += 1
                # Type accuracy
                if pred_piece == gt_piece:
                    type_correct += 1

    total_occ = max(total_occupied_gt, 1)
    return {
        "empty_acc": empty_correct / 64.0,
        "presence_acc": presence_correct / 64.0,
        "color_acc": color_correct / total_occ,
        "type_acc": type_correct / total_occ,
        "exact_match": pred_board.board_fen() == gt_board.board_fen(),
    }


def print_board_comparison(predicted_fen: str | None, ground_truth_fen: str, label: str = ""):
    """Print two boards side by side for visual comparison."""
    if label:
        print(f"\n--- {label} ---")

    pred_board = (
        chess.Board(predicted_fen + " w - - 0 1") if predicted_fen else chess.Board(fen=None)
    )
    gt_board = chess.Board(
        ground_truth_fen if " " in ground_truth_fen else ground_truth_fen + " w - - 0 1"
    )

    print("  Predicted:                  Ground Truth:")
    pred_lines = str(pred_board).split("\n")
    gt_lines = str(gt_board).split("\n")
    for pl, gl in zip(pred_lines, gt_lines):
        print(f"  {pl}        {gl}")
    print()
