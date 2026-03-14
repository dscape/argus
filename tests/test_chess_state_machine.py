"""Tests for the chess state machine."""

import chess
import pytest
import torch

from argus.chess.move_vocabulary import NO_MOVE_IDX, VOCAB_SIZE, get_vocabulary
from argus.chess.state_machine import GameStateMachine


@pytest.fixture
def sm() -> GameStateMachine:
    return GameStateMachine()


class TestGameStateMachine:
    def test_initial_position(self, sm: GameStateMachine) -> None:
        assert sm.get_fen() == chess.STARTING_FEN
        assert sm.move_count == 0
        assert sm.fullmove_number == 1
        assert sm.turn == chess.WHITE

    def test_push_legal_move(self, sm: GameStateMachine) -> None:
        assert sm.push_move("e2e4") is True
        assert sm.move_count == 1
        assert sm.turn == chess.BLACK

    def test_push_illegal_move(self, sm: GameStateMachine) -> None:
        assert sm.push_move("e2e5") is False  # Can't jump 3 squares
        assert sm.move_count == 0

    def test_push_invalid_uci(self, sm: GameStateMachine) -> None:
        assert sm.push_move("xyz") is False
        assert sm.push_move("") is False

    def test_is_legal(self, sm: GameStateMachine) -> None:
        assert sm.is_legal("e2e4") is True
        assert sm.is_legal("e2e5") is False
        assert sm.is_legal("e7e5") is False  # Black's move, but it's White's turn

    def test_legal_mask_shape(self, sm: GameStateMachine) -> None:
        mask = sm.get_legal_mask()
        assert mask.shape == (VOCAB_SIZE,)
        assert mask.dtype == torch.bool

    def test_legal_mask_no_move_always_true(self, sm: GameStateMachine) -> None:
        mask = sm.get_legal_mask()
        assert mask[NO_MOVE_IDX].item() is True

    def test_legal_mask_covers_all_legal_moves(self, sm: GameStateMachine) -> None:
        vocab = get_vocabulary()
        mask = sm.get_legal_mask()
        for uci in sm.get_legal_moves_uci():
            idx = vocab.uci_to_index(uci)
            assert mask[idx].item() is True, f"Legal move {uci} not in mask"

    def test_legal_mask_blocks_illegal(self, sm: GameStateMachine) -> None:
        vocab = get_vocabulary()
        mask = sm.get_legal_mask()
        # e7e5 is Black's move — should be blocked in starting position
        idx = vocab.uci_to_index("e7e5")
        assert mask[idx].item() is False

    def test_game_sequence(self, sm: GameStateMachine) -> None:
        """Play Scholar's Mate and verify game over."""
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
        for uci in moves:
            assert sm.push_move(uci) is True
        assert sm.is_game_over() is True
        assert sm.result() == "1-0"

    def test_to_pgn(self, sm: GameStateMachine) -> None:
        sm.push_move("e2e4")
        sm.push_move("e7e5")
        pgn = sm.to_pgn()
        assert "1. e4 e5" in pgn
        assert "[Result" in pgn

    def test_custom_start_fen(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        sm = GameStateMachine(fen=fen)
        assert sm.turn == chess.BLACK
        assert sm.push_move("e7e5") is True

    def test_copy(self, sm: GameStateMachine) -> None:
        sm.push_move("e2e4")
        copy = sm.copy()
        copy.push_move("e7e5")
        assert sm.move_count == 1  # Original unchanged
        assert copy.move_count == 2

    def test_get_legal_moves_uci(self, sm: GameStateMachine) -> None:
        legal = sm.get_legal_moves_uci()
        assert len(legal) == 20  # 20 legal opening moves
        assert "e2e4" in legal
        assert "d2d4" in legal

    def test_result_in_progress(self, sm: GameStateMachine) -> None:
        assert sm.result() == "*"

    def test_get_best_legal_alternative(self, sm: GameStateMachine) -> None:
        vocab = get_vocabulary()
        logits = torch.full((VOCAB_SIZE,), -10.0)
        # Set e2e4 to have high logit
        logits[vocab.uci_to_index("e2e4")] = 10.0
        best = sm.get_best_legal_alternative(logits)
        assert best == "e2e4"
