"""Board-state targets for dense supervision."""

from __future__ import annotations

import chess
import torch

SQUARE_CLASS_NAMES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
SQUARE_CLASS_TO_INDEX = {name: index for index, name in enumerate(SQUARE_CLASS_NAMES)}
NUM_SQUARE_CLASSES = len(SQUARE_CLASS_NAMES)


def fen_to_square_targets(fen: str, *, board_flipped: bool = False) -> torch.Tensor:
    """Convert a FEN string to 64 piece-on-square class targets.

    Targets are ordered row-major from top-left to bottom-right in the rendered board image.
    For standard orientation this matches A8..H8, A7..H7, ..., A1..H1.
    When ``board_flipped`` is true, the targets are rotated 180 degrees to match a
    black-side perspective render.
    """
    if " " not in fen:
        fen = f"{fen} w - - 0 1"
    board = chess.Board(fen)
    targets = []
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            symbol = piece.symbol() if piece is not None else "empty"
            targets.append(SQUARE_CLASS_TO_INDEX[symbol])
    if board_flipped:
        targets.reverse()
    return torch.tensor(targets, dtype=torch.long)
