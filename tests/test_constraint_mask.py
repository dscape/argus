"""Tests for constraint mask generation and application."""

import chess
import pytest
import torch

from argus.chess.constraint_mask import (
    apply_constraint_mask,
    get_legal_mask,
    get_legal_mask_batch,
)
from argus.chess.move_vocabulary import NO_MOVE_IDX, UNKNOWN_IDX, VOCAB_SIZE, get_vocabulary


class TestGetLegalMask:
    def test_starting_position(self) -> None:
        board = chess.Board()
        mask = get_legal_mask(board)
        assert mask.shape == (VOCAB_SIZE,)
        # 20 legal moves + NO_MOVE = 21 True values
        assert mask.sum().item() == 21

    def test_no_move_always_legal(self) -> None:
        board = chess.Board()
        mask = get_legal_mask(board)
        assert mask[NO_MOVE_IDX].item() is True

    def test_unknown_always_illegal(self) -> None:
        board = chess.Board()
        mask = get_legal_mask(board)
        assert mask[UNKNOWN_IDX].item() is False

    def test_checkmate_position(self) -> None:
        """In checkmate, only NO_MOVE should be legal."""
        # Scholar's Mate final position
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push(chess.Move.from_uci(uci))
        mask = get_legal_mask(board)
        # Game is over — no legal chess moves, only NO_MOVE
        assert mask.sum().item() == 1
        assert mask[NO_MOVE_IDX].item() is True

    def test_batch(self) -> None:
        boards = [chess.Board(), chess.Board("8/8/8/8/8/8/6k1/4K2R w K - 0 1")]
        masks = get_legal_mask_batch(boards)
        assert masks.shape == (2, VOCAB_SIZE)


class TestApplyConstraintMask:
    def test_illegal_moves_masked(self) -> None:
        logits = torch.randn(VOCAB_SIZE)
        mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        mask[0] = True  # Only one legal move
        mask[NO_MOVE_IDX] = True

        masked = apply_constraint_mask(logits, mask)
        # Illegal positions should be -1e9
        assert masked[1].item() == pytest.approx(-1e9)
        # Legal positions should be unchanged
        assert masked[0].item() == logits[0].item()

    def test_softmax_after_mask(self) -> None:
        logits = torch.randn(VOCAB_SIZE)
        mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        mask[0] = True
        mask[NO_MOVE_IDX] = True

        masked = apply_constraint_mask(logits, mask)
        probs = torch.softmax(masked, dim=-1)

        # Probability should be concentrated on legal moves
        legal_prob = probs[0].item() + probs[NO_MOVE_IDX].item()
        assert legal_prob == pytest.approx(1.0, abs=1e-5)
