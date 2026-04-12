"""Shared board-state label utilities for overlay and physical pipelines."""

from __future__ import annotations

SQUARE_CLASS_NAMES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
NUM_SQUARE_CLASSES = len(SQUARE_CLASS_NAMES)

_PIECE_TO_CLASS = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}


def fen_to_square_labels(fen: str) -> list[list[int]]:
    """Convert piece-placement FEN to an 8×8 grid of square-class indices."""
    piece_placement = fen.split(" ", 1)[0]
    ranks = piece_placement.split("/")
    assert len(ranks) == 8, f"Expected 8 ranks, got {len(ranks)}: {fen}"

    grid: list[list[int]] = []
    for rank_str in ranks:
        row: list[int] = []
        for char in rank_str:
            if char.isdigit():
                row.extend([0] * int(char))
                continue

            class_index = _PIECE_TO_CLASS.get(char)
            assert class_index is not None, f"Unknown piece char: {char}"
            row.append(class_index)

        assert len(row) == 8, f"Rank has {len(row)} squares: {rank_str}"
        grid.append(row)

    return grid
