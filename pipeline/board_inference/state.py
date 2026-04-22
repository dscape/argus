"""Board representation, logit scoring, and FEN round-trips (python-chess)."""

from __future__ import annotations

import math

import chess
import numpy as np

from pipeline.shared import SQUARE_CLASS_NAMES
from pipeline.shared.board_constraints import constrained_board_class_ids
from pipeline.shared.board_tracking import board_to_class_ids

# Matches ``SQUARE_CLASS_NAMES[1:7]`` and ``SQUARE_CLASS_NAMES[7:13]``.
_PIECE_TYPES: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Return log-softmax of shape (64, C) (stable)."""
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=1, keepdims=True)
    ex = np.exp(x - m)
    s = np.sum(ex, axis=1, keepdims=True)
    return (x - m) - np.log(s + 1e-30)


def class_id_to_symbol(class_id: int) -> str:
    if 0 <= class_id < len(SQUARE_CLASS_NAMES):
        return SQUARE_CLASS_NAMES[class_id]
    raise ValueError(f"class_id out of range: {class_id}")


def class_id_to_python_chess_piece(class_id: int) -> chess.Piece | None:
    if class_id == 0:
        return None
    if 1 <= class_id <= 6:
        return chess.Piece(_PIECE_TYPES[class_id - 1], chess.WHITE)
    if 7 <= class_id <= 12:
        return chess.Piece(_PIECE_TYPES[class_id - 7], chess.BLACK)
    raise ValueError(f"Unknown class_id: {class_id}")


def class_ids_to_piece_placement_fen(class_ids: list[int]) -> str:
    if len(class_ids) != 64:
        raise ValueError(f"Expected 64 class ids, got {len(class_ids)}")

    ranks: list[str] = []
    for row in range(8):
        empty_run = 0
        parts: list[str] = []
        for col in range(8):
            c = class_ids[row * 8 + col]
            name = class_id_to_symbol(c)
            if name == "empty":
                empty_run += 1
                continue
            if empty_run:
                parts.append(str(empty_run))
                empty_run = 0
            parts.append(name)
        if empty_run:
            parts.append(str(empty_run))
        ranks.append("".join(parts) or "8")
    return "/".join(ranks)


def board_from_class_ids(
    class_ids: list[int],
    *,
    turn: chess.Color = chess.WHITE,
) -> chess.Board:
    """Construct a :class:`chess.Board` from 64 per-square class ids (Argus order)."""
    piece_fen = class_ids_to_piece_placement_fen(class_ids)
    # No castling / e.p. in snapshot mode; for tracking, caller can repair via ``push``.
    # ``chess.Color`` is ``bool`` in python-chess (WHITE=True, BLACK=False).
    turn_char = "w" if turn else "b"
    return chess.Board(f"{piece_fen} {turn_char} - - 0 1")


def numpy_logits_to_constrained_class_ids(logits: np.ndarray) -> list[int]:
    import torch  # local: keeps ``state`` import graph light

    t = torch.as_tensor(logits, dtype=torch.float32)
    return [int(x) for x in constrained_board_class_ids(t).tolist()]


# Heavy penalty (not -inf) so we can still order impossible boards in argmax.
LOGIT_SCORE_PENALTY: float = -1e3


def score_board(board: chess.Board, logits: np.ndarray) -> float:
    """Sum per-square log-probabilities of observed pieces, plus a penalty for invalid shapshots.

    ``logits`` is raw per-square pre-softmax, shape (64, 13).
    """
    logp = log_softmax_rows(np.asarray(logits, dtype=np.float64))
    try:
        class_ids = board_to_class_ids(board)
    except (AssertionError, ValueError):
        return float(13 * 64) * LOGIT_SCORE_PENALTY
    if len(class_ids) != 64:
        return float(13 * 64) * LOGIT_SCORE_PENALTY
    idx = np.arange(64, dtype=np.intp)
    base = float(np.sum(logp[idx, class_ids]))
    from .constraints import hard_constraint_penalty

    return base + hard_constraint_penalty(class_ids)


class BoardState:
    def __init__(self, board: chess.Board, score: float):
        self.board = board
        self.score = score

    @property
    def confidence(self) -> float:
        """A bounded proxy for the composite score (higher = better)."""
        return 1.0 / (1.0 + math.exp(-self.score / 64.0))

    @property
    def fen(self) -> str:
        return self.board.fen()

    def __repr__(self) -> str:
        return f"BoardState(score={self.score:.4f}, fen={self.board.fen()!r})"
