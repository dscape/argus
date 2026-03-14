"""Tests for move vocabulary."""

import chess
import pytest

from argus.chess.move_vocabulary import (
    NO_MOVE_IDX,
    UNKNOWN_IDX,
    VOCAB_SIZE,
    MoveVocabulary,
    get_vocabulary,
)


@pytest.fixture
def vocab() -> MoveVocabulary:
    return MoveVocabulary()


class TestMoveVocabulary:
    def test_vocab_size(self, vocab: MoveVocabulary) -> None:
        assert vocab.size == VOCAB_SIZE
        assert vocab.size == 1970

    def test_num_moves(self, vocab: MoveVocabulary) -> None:
        assert vocab.num_moves == 1968

    def test_bijectivity(self, vocab: MoveVocabulary) -> None:
        """Every index maps to a unique UCI string and back."""
        seen_uci: set[str] = set()
        for idx in range(VOCAB_SIZE):
            uci = vocab.index_to_uci(idx)
            assert uci not in seen_uci, f"Duplicate UCI: {uci}"
            seen_uci.add(uci)
            assert vocab.uci_to_index(uci) == idx

    def test_special_tokens(self, vocab: MoveVocabulary) -> None:
        assert vocab.index_to_uci(NO_MOVE_IDX) == "<no_move>"
        assert vocab.index_to_uci(UNKNOWN_IDX) == "<unknown>"
        assert vocab.uci_to_index("<no_move>") == NO_MOVE_IDX
        assert vocab.uci_to_index("<unknown>") == UNKNOWN_IDX

    def test_common_moves_present(self, vocab: MoveVocabulary) -> None:
        """Standard opening moves should be in the vocabulary."""
        common_moves = ["e2e4", "d2d4", "g1f3", "b1c3", "e7e5", "d7d5"]
        for uci in common_moves:
            assert vocab.contains(uci), f"Missing common move: {uci}"

    def test_promotions_present(self, vocab: MoveVocabulary) -> None:
        """Promotion moves should be in the vocabulary."""
        promo_moves = ["a7a8q", "a7a8r", "a7a8b", "a7a8n", "h2h1q", "g7f8n"]
        for uci in promo_moves:
            assert vocab.contains(uci), f"Missing promotion: {uci}"

    def test_no_self_moves(self, vocab: MoveVocabulary) -> None:
        """A square to itself should not be in the vocabulary."""
        for sq in chess.SQUARES:
            name = chess.square_name(sq)
            assert not vocab.contains(f"{name}{name}")

    def test_all_legal_opening_moves_covered(self, vocab: MoveVocabulary) -> None:
        """Every legal move from the starting position should be in vocab."""
        board = chess.Board()
        for move in board.legal_moves:
            assert vocab.contains(move.uci()), f"Missing legal move: {move.uci()}"

    def test_all_legal_moves_in_random_positions(self, vocab: MoveVocabulary) -> None:
        """Legal moves from several positions should all be in vocab."""
        # Play through a short game and check at each position
        board = chess.Board()
        moves_to_play = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
        for uci in moves_to_play:
            board.push(chess.Move.from_uci(uci))
            for legal in board.legal_moves:
                assert vocab.contains(legal.uci()), (
                    f"Missing legal move {legal.uci()} from FEN {board.fen()}"
                )

    def test_singleton(self) -> None:
        v1 = get_vocabulary()
        v2 = get_vocabulary()
        assert v1 is v2

    def test_deterministic(self) -> None:
        """Two independent instances produce the same mapping."""
        v1 = MoveVocabulary()
        v2 = MoveVocabulary()
        for idx in range(VOCAB_SIZE):
            assert v1.index_to_uci(idx) == v2.index_to_uci(idx)
