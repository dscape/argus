"""Generate legal move masks from chess board state.

The constraint mask is a binary vector over the move vocabulary that
zeros out illegal moves. This is applied to model logits before softmax,
making it architecturally impossible to predict an illegal move.
"""

from __future__ import annotations

import chess
import torch

from argus.chess.move_vocabulary import NO_MOVE_IDX, VOCAB_SIZE, get_vocabulary


def get_legal_mask(board: chess.Board) -> torch.Tensor:
    """Generate a binary mask over the move vocabulary for legal moves.

    Args:
        board: Current chess board position.

    Returns:
        Boolean tensor of shape (VOCAB_SIZE,) where True = legal.
        NO_MOVE is always legal. UNKNOWN is always illegal.
    """
    vocab = get_vocabulary()
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)

    # NO_MOVE is always legal (most frames have no move)
    mask[NO_MOVE_IDX] = True

    for move in board.legal_moves:
        uci = move.uci()
        if vocab.contains(uci):
            mask[vocab.uci_to_index(uci)] = True

    return mask


def get_legal_mask_batch(boards: list[chess.Board]) -> torch.Tensor:
    """Generate legal move masks for a batch of boards.

    Args:
        boards: List of chess board positions.

    Returns:
        Boolean tensor of shape (len(boards), VOCAB_SIZE).
    """
    return torch.stack([get_legal_mask(b) for b in boards])


def apply_constraint_mask(
    logits: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Apply legal move mask to logits, setting illegal moves to fill_value.

    Args:
        logits: Raw model output, shape (..., VOCAB_SIZE).
        mask: Boolean mask, shape (..., VOCAB_SIZE). True = legal.
        fill_value: Value to assign to illegal moves (large negative for softmax).

    Returns:
        Masked logits with same shape as input.
    """
    return logits.masked_fill(~mask, fill_value)
