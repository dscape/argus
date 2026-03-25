"""Strategy 4: Frame differencing + chess-legal move tracking.

Don't read the absolute position — track CHANGES between consecutive
frames and validate against legal chess moves. Requires bootstrapping
from the starting position.
"""

import chess
import cv2
import numpy as np
from shared import (
    _inner_crop,
    classify_empty,
    crop_squares,
    extract_frame,
    find_board_in_frame,
)

STRATEGY_NAME = "S4: Move Track"


def _compute_square_similarity(sq1: np.ndarray, sq2: np.ndarray) -> float:
    """Compute pixel MSE between two square images (normalized)."""
    inner1 = _inner_crop(sq1, 0.6)
    inner2 = _inner_crop(sq2, 0.6)
    if inner1.size == 0 or inner2.size == 0:
        return 0.0

    # Resize to same size
    size = 32
    r1 = cv2.resize(inner1, (size, size)).astype(float)
    r2 = cv2.resize(inner2, (size, size)).astype(float)
    return float(np.mean((r1 - r2) ** 2))


class MoveTracker:
    """Track moves frame-by-frame from a known starting position."""

    def __init__(self):
        self.board: chess.Board | None = None
        self.prev_squares: list[list[np.ndarray]] | None = None
        self.prev_grid: tuple | None = None
        self.flipped: bool = False

    def bootstrap(self, video_path: str) -> bool:
        """Find starting position and initialize tracking.

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

            n_empty = sum(empty_mask[r][c] for r in range(8) for c in range(8))
            if n_empty < 28 or n_empty > 36:
                continue

            # Check if middle rows are empty (starting position pattern)
            middle_empty = sum(empty_mask[r][c] for r in range(2, 6) for c in range(8))
            if middle_empty < 28:
                continue

            self.board = chess.Board()  # Standard starting position
            self.prev_squares = squares
            self.prev_grid = grid
            return True

        return False

    def track_move(self, frame: np.ndarray) -> str | None:
        """Detect if a move occurred between previous frame and this one.

        Returns the current board FEN.
        """
        if self.board is None or self.prev_squares is None:
            return None

        grid = find_board_in_frame(frame)
        if grid is None:
            return self.board.board_fen() if self.board else None

        v_lines, h_lines, sq_size = grid
        curr_squares = crop_squares(frame, v_lines, h_lines)

        # Compute per-square change magnitude
        changes = np.zeros((8, 8))
        for r in range(8):
            for c in range(8):
                changes[r, c] = _compute_square_similarity(
                    self.prev_squares[r][c], curr_squares[r][c]
                )

        # Find changed squares using Otsu on the change values
        flat = changes.flatten()
        flat_u8 = np.clip(flat / max(flat.max(), 1) * 255, 0, 255).astype(np.uint8)
        if flat.max() - flat.min() < 10:
            # No significant changes
            self.prev_squares = curr_squares
            return self.board.board_fen()

        thresh_val, _ = cv2.threshold(flat_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = thresh_val / 255.0 * flat.max()

        changed_squares: list[tuple[int, int]] = []
        for r in range(8):
            for c in range(8):
                if changes[r, c] > threshold:
                    changed_squares.append((r, c))

        if len(changed_squares) < 2:
            # No move detected
            self.prev_squares = curr_squares
            return self.board.board_fen()

        # Convert display coordinates to chess squares
        def display_to_chess_sq(r: int, c: int) -> int:
            if not self.flipped:
                return chess.square(c, 7 - r)
            else:
                return chess.square(7 - c, r)

        changed_chess_sqs = {display_to_chess_sq(r, c) for r, c in changed_squares}

        # Find the legal move that best matches the changed squares
        best_move = None
        best_overlap = 0

        for move in self.board.legal_moves:
            move_squares = {move.from_square, move.to_square}

            # Castling involves rook movement too
            if self.board.is_castling(move):
                if self.board.is_kingside_castling(move):
                    rook_from = chess.square(7, chess.square_rank(move.from_square))
                    rook_to = chess.square(5, chess.square_rank(move.from_square))
                else:
                    rook_from = chess.square(0, chess.square_rank(move.from_square))
                    rook_to = chess.square(3, chess.square_rank(move.from_square))
                move_squares.update({rook_from, rook_to})

            # En passant: captured pawn square also changes
            if self.board.is_en_passant(move):
                captured_sq = chess.square(
                    chess.square_file(move.to_square), chess.square_rank(move.from_square)
                )
                move_squares.add(captured_sq)

            overlap = len(move_squares & changed_chess_sqs)
            if overlap > best_overlap:
                best_overlap = overlap
                best_move = move

        if best_move is not None and best_overlap >= 2:
            self.board.push(best_move)

        self.prev_squares = curr_squares
        return self.board.board_fen()


# Module-level state
_tracker: MoveTracker | None = None


def read_position(frame: np.ndarray, video_path: str | None = None) -> str | None:
    """Track position via move detection.

    First call must provide video_path for bootstrapping.
    Subsequent calls track moves from the previous frame.
    """
    global _tracker
    if _tracker is None or video_path is not None:
        _tracker = MoveTracker()
        if video_path:
            if not _tracker.bootstrap(video_path):
                from strategy_1_area_ranking import read_position as fallback

                return fallback(frame)

    return _tracker.track_move(frame)
