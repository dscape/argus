"""Per-square (64×13) logits → globally consistent :class:`chess.Board` (optional time tracking)."""

from __future__ import annotations

from .solver import generate_candidates, infer_board
from .state import (
    BoardState,
    class_ids_to_piece_placement_fen,
    log_softmax_rows,
    score_board,
)
from .tracker import NO_MOVE_BONUS, Tracker, track_boards

__all__ = [
    "NO_MOVE_BONUS",
    "BoardState",
    "Tracker",
    "class_ids_to_piece_placement_fen",
    "generate_candidates",
    "infer_board",
    "log_softmax_rows",
    "score_board",
    "track_boards",
]
