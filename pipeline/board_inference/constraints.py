"""Hard/soft plausibility checks for Argus 64×13 square labels."""

from __future__ import annotations

import collections

import chess

# Same back ranks as ``pipeline.shared.board_constraints`` (rank-8 and rank-1 a–h in Argus order).
_BACK_RANK_CLASS_IDS: tuple[int, ...] = tuple(range(8)) + tuple(range(56, 64))
_WHITE_PAWN = 1
_BLACK_PAWN = 7
_WHITE_KING = 6
_BLACK_KING = 12
_NO_BONUS: float = 0.0


def argus_index_to_square(square_index: int) -> int:
    """Map Argus 0..63 index (FEN top rank first) to a python-chess ``Square``."""
    col = square_index % 8
    row = square_index // 8
    return chess.square(col, 7 - row)


def count_pieces_by_class(class_ids: list[int]) -> collections.Counter[int]:
    return collections.Counter(class_ids)


def kings_too_close(class_ids: list[int]) -> bool:
    wk: int | None = None
    bk: int | None = None
    for i, c in enumerate(class_ids):
        if c == _WHITE_KING:
            if wk is not None:
                return True
            wk = i
        elif c == _BLACK_KING:
            if bk is not None:
                return True
            bk = i
    if wk is None or bk is None:
        return True
    return bool(chess.square_distance(argus_index_to_square(wk), argus_index_to_square(bk)) < 2)


def satisfies_hard_class_constraints(class_ids: list[int]) -> bool:
    if len(class_ids) != 64:
        return False
    counts = count_pieces_by_class(class_ids)
    if counts[_WHITE_KING] != 1 or counts[_BLACK_KING] != 1:
        return False
    if counts[_WHITE_PAWN] > 8 or counts[_BLACK_PAWN] > 8:
        return False
    for i in _BACK_RANK_CLASS_IDS:
        if class_ids[i] == _WHITE_PAWN or class_ids[i] == _BLACK_PAWN:
            return False
    if kings_too_close(class_ids):
        return False
    return True


def hard_constraint_penalty(
    class_ids: list[int],
) -> float:
    if not satisfies_hard_class_constraints(class_ids):
        return 10.0 * 64.0
    return _NO_BONUS
