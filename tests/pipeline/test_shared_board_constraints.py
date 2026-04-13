from __future__ import annotations

import torch
from pipeline.shared.board_constraints import constrained_board_class_ids

NUM_CLASSES = 13
WHITE_KING = 6
BLACK_KING = 12
WHITE_PAWN = 1
BLACK_PAWN = 7


def _empty_board_logits() -> torch.Tensor:
    logits = torch.zeros((64, NUM_CLASSES), dtype=torch.float32)
    logits[:, 0] = 1.0
    return logits


def test_constrained_board_class_ids_removes_back_rank_pawns() -> None:
    logits = _empty_board_logits()
    logits[0, WHITE_PAWN] = 5.0
    logits[0, 3] = 4.0
    logits[4, BLACK_KING] = 6.0
    logits[60, WHITE_KING] = 6.0

    class_ids = constrained_board_class_ids(logits)

    assert int(class_ids[0].item()) == 3


def test_constrained_board_class_ids_inserts_missing_white_king() -> None:
    logits = _empty_board_logits()
    logits[4, BLACK_KING] = 6.0
    logits[60, 4] = 3.0
    logits[60, WHITE_KING] = 2.5

    class_ids = constrained_board_class_ids(logits)

    assert int((class_ids == WHITE_KING).sum().item()) == 1
    assert int(class_ids[60].item()) == WHITE_KING


def test_constrained_board_class_ids_removes_extra_black_kings() -> None:
    logits = _empty_board_logits()
    logits[4, BLACK_KING] = 6.0
    logits[12, BLACK_KING] = 5.5
    logits[12, 10] = 5.0
    logits[60, WHITE_KING] = 6.0

    class_ids = constrained_board_class_ids(logits)

    assert int((class_ids == BLACK_KING).sum().item()) == 1
    assert int(class_ids[4].item()) == BLACK_KING
    assert int(class_ids[12].item()) == 10


def test_constrained_board_class_ids_supports_batches() -> None:
    logits = torch.stack([_empty_board_logits(), _empty_board_logits()], dim=0)
    logits[:, 4, BLACK_KING] = 6.0
    logits[:, 60, WHITE_KING] = 6.0

    class_ids = constrained_board_class_ids(logits)

    assert class_ids.shape == (2, 64)
