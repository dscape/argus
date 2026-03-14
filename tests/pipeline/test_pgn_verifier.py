"""Tests for pipeline.match.pgn_verifier."""

import pytest
from pipeline.match.pgn_verifier import _parse_moves, verify_pgn


# Reusable PGN strings
ITALIAN_GAME = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. Qb3 Nce7 11. O-O O-O 12. Rfe1 c6 13. Ne4 Nf5 14. a4 b6 15. Nc5 Qd6 1/2-1/2"
SCHOLARS_MATE = "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0"


# ── _parse_moves ─────────────────────────────────────────────


class TestParseMoves:
    def test_parses_movetext(self):
        moves = _parse_moves(SCHOLARS_MATE)
        assert len(moves) == 7  # 4 white + 3 black half-moves
        assert moves[0] == "e2e4"
        assert moves[1] == "e7e5"

    def test_returns_uci(self):
        moves = _parse_moves("1. e4 e5 2. Nf3 Nc6")
        assert moves == ["e2e4", "e7e5", "g1f3", "b8c6"]

    def test_empty_string(self):
        assert _parse_moves("") == []

    def test_none_input(self):
        assert _parse_moves(None) == []

    def test_invalid_pgn(self):
        assert _parse_moves("this is not pgn") == []

    def test_pgn_with_headers(self):
        pgn = '[Event "Test"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0'
        moves = _parse_moves(pgn)
        assert len(moves) == 4
        assert moves[0] == "e2e4"


# ── verify_pgn ───────────────────────────────────────────────


class TestVerifyPgn:
    def test_identical_pgns(self):
        score = verify_pgn(ITALIAN_GAME, ITALIAN_GAME)
        assert score == 1.0

    def test_identical_short_game(self):
        score = verify_pgn(SCHOLARS_MATE, SCHOLARS_MATE)
        assert score == 1.0

    def test_completely_different(self):
        other = "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 1/2-1/2"
        score = verify_pgn(ITALIAN_GAME, other)
        assert score == 0.0

    def test_first_move_differs(self):
        game_a = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0"
        game_b = "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 1-0"
        assert verify_pgn(game_a, game_b) == 0.0

    def test_diverges_after_five_moves(self):
        game_a = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0"
        game_b = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Bc5 6. c3 d6 1-0"
        score = verify_pgn(game_a, game_b)
        # 8 out of 12 moves match (first 8 half-moves identical, diverges at move 5 for black)
        assert 0.0 < score < 1.0

    def test_empty_game_pgn(self):
        assert verify_pgn("", SCHOLARS_MATE) == 0.0

    def test_empty_video_pgn(self):
        assert verify_pgn(SCHOLARS_MATE, "") == 0.0

    def test_both_empty(self):
        assert verify_pgn("", "") == 0.0

    def test_long_matching_prefix_gets_high_score(self):
        """15+ matching moves should give score of 1.0."""
        score = verify_pgn(ITALIAN_GAME, ITALIAN_GAME, num_moves=15)
        assert score == 1.0

    def test_custom_num_moves(self):
        score = verify_pgn(ITALIAN_GAME, ITALIAN_GAME, num_moves=5)
        assert score == 1.0

    def test_pgn_with_headers_vs_movetext(self):
        """Video PGN may have headers; game PGN is movetext-only."""
        with_headers = '[Event "Test"]\n[Result "1-0"]\n\n' + SCHOLARS_MATE
        score = verify_pgn(SCHOLARS_MATE, with_headers)
        assert score == 1.0
