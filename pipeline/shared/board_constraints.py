"""Shared board-state postprocessing constraints."""

from __future__ import annotations

import torch

_WHITE_PAWN_CLASS = 1
_BLACK_PAWN_CLASS = 7
_WHITE_KING_CLASS = 6
_BLACK_KING_CLASS = 12
_BACK_RANK_INDICES = tuple(range(8)) + tuple(range(56, 64))
_PAWN_CLASSES = {_WHITE_PAWN_CLASS, _BLACK_PAWN_CLASS}


def constrained_board_class_ids(square_logits: torch.Tensor) -> torch.Tensor:
    """Return board class ids after lightweight chess-aware postprocessing.

    Current constraints are intentionally conservative:
    - pawns cannot remain on the first or eighth rank
    - each color must have at least one king

    The function accepts either `(64, C)` logits for one board or `(B, 64, C)` for a batch.
    """
    if square_logits.ndim == 2:
        return _constrained_single_board_class_ids(square_logits)
    if square_logits.ndim == 3:
        return torch.stack(
            [_constrained_single_board_class_ids(board_logits) for board_logits in square_logits],
            dim=0,
        )
    raise ValueError(f"Expected board logits with ndim 2 or 3, got {square_logits.ndim}")



def _constrained_single_board_class_ids(square_logits: torch.Tensor) -> torch.Tensor:
    if square_logits.shape[0] != 64:
        raise ValueError(f"Expected 64 board squares, got {square_logits.shape[0]}")
    if square_logits.shape[1] <= _BLACK_KING_CLASS:
        raise ValueError(
            f"Expected at least {_BLACK_KING_CLASS + 1} classes, got {square_logits.shape[1]}"
        )

    class_ids = square_logits.argmax(dim=1).clone()

    for square_index in _BACK_RANK_INDICES:
        if int(class_ids[square_index].item()) in _PAWN_CLASSES:
            class_ids[square_index] = _best_allowed_class_id(
                square_logits[square_index],
                forbidden_class_ids=_PAWN_CLASSES,
            )

    class_ids = _ensure_king_present(
        square_logits,
        class_ids,
        king_class_id=_WHITE_KING_CLASS,
        opposing_king_class_id=_BLACK_KING_CLASS,
    )
    class_ids = _ensure_king_present(
        square_logits,
        class_ids,
        king_class_id=_BLACK_KING_CLASS,
        opposing_king_class_id=_WHITE_KING_CLASS,
    )
    return class_ids



def _ensure_king_present(
    square_logits: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    king_class_id: int,
    opposing_king_class_id: int,
) -> torch.Tensor:
    if bool((class_ids == king_class_id).any()):
        return class_ids

    current_scores = square_logits.gather(1, class_ids.unsqueeze(1)).squeeze(1)
    king_scores = square_logits[:, king_class_id]
    gains = king_scores - current_scores
    gains = gains.masked_fill(class_ids == opposing_king_class_id, float("-inf"))
    replacement_index = int(gains.argmax().item())
    class_ids[replacement_index] = king_class_id
    return class_ids



def _best_allowed_class_id(
    square_logits: torch.Tensor,
    *,
    forbidden_class_ids: set[int],
) -> torch.Tensor:
    masked_logits = square_logits.clone()
    forbidden_indices = torch.tensor(
        sorted(forbidden_class_ids),
        dtype=torch.long,
        device=square_logits.device,
    )
    masked_logits[forbidden_indices] = float("-inf")
    return masked_logits.argmax(dim=0)
