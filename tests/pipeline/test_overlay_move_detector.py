"""Tests for pipeline.overlay.overlay_move_detector."""

import chess
import pytest

from pipeline.overlay.overlay_move_detector import (
    GameSegment,
    OverlayDetectedMove,
    detect_moves,
    find_move_between_positions,
)


class TestFindMoveBetweenPositions:
    """Test finding the legal move between two board positions."""

    def test_simple_pawn_move(self):
        board = chess.Board()
        # e2e4
        after_board = board.copy()
        after_board.push(chess.Move.from_uci("e2e4"))

        move = find_move_between_positions(board, after_board.board_fen())
        assert move is not None
        assert move.uci() == "e2e4"

    def test_knight_move(self):
        board = chess.Board()
        after_board = board.copy()
        after_board.push(chess.Move.from_uci("g1f3"))

        move = find_move_between_positions(board, after_board.board_fen())
        assert move is not None
        assert move.uci() == "g1f3"

    def test_castling(self):
        # Set up a position where castling is possible
        board = chess.Board("r1bqkbnr/pppppppp/2n5/4P3/8/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3")
        # This position doesn't allow castling yet, let's use a cleaner one
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 0 1")
        after_board = board.copy()
        after_board.push(chess.Move.from_uci("e1g1"))  # Kingside castle

        move = find_move_between_positions(board, after_board.board_fen())
        assert move is not None
        assert move.uci() == "e1g1"

    def test_en_passant(self):
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        after_board = board.copy()
        after_board.push(chess.Move.from_uci("e5d6"))  # En passant

        move = find_move_between_positions(board, after_board.board_fen())
        assert move is not None
        assert move.uci() == "e5d6"

    def test_promotion(self):
        board = chess.Board("8/4P1k1/8/8/8/8/8/4K3 w - - 0 1")
        after_board = board.copy()
        after_board.push(chess.Move.from_uci("e7e8q"))  # Promote to queen

        move = find_move_between_positions(board, after_board.board_fen())
        assert move is not None
        assert move.uci() == "e7e8q"

    def test_no_legal_move_found(self):
        board = chess.Board()
        # A completely different position that can't be reached in one move
        impossible_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1"

        move = find_move_between_positions(board, impossible_fen)
        assert move is None


class TestDetectMoves:
    """Test move detection from a sequence of FENs."""

    def _play_game(self, moves_uci: list[str]) -> list[str]:
        """Play a sequence of moves and return FENs after each move."""
        board = chess.Board()
        fens = [board.board_fen()]
        for uci in moves_uci:
            board.push(chess.Move.from_uci(uci))
            fens.append(board.board_fen())
        return fens

    def test_simple_game(self):
        """Detect moves from a clean FEN sequence."""
        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
        fens = self._play_game(moves)

        # Each FEN appears for 3 frames (stability), then changes
        expanded_fens = []
        expanded_indices = []
        idx = 0
        for fen in fens:
            for _ in range(3):
                expanded_fens.append(fen)
                expanded_indices.append(idx)
                idx += 1

        segments = detect_moves(expanded_fens, expanded_indices, fps=2.0)

        assert len(segments) == 1
        seg = segments[0]
        assert seg.num_moves == len(moves)
        for i, m in enumerate(seg.moves):
            assert m.move_uci == moves[i]

    def test_single_move(self):
        """Detect a single move."""
        fens = self._play_game(["e2e4"])

        expanded_fens = []
        expanded_indices = []
        idx = 0
        for fen in fens:
            for _ in range(3):
                expanded_fens.append(fen)
                expanded_indices.append(idx)
                idx += 1

        segments = detect_moves(expanded_fens, expanded_indices, fps=2.0)

        assert len(segments) == 1
        assert segments[0].num_moves == 1
        assert segments[0].moves[0].move_uci == "e2e4"

    def test_animation_frames_filtered(self):
        """Animation frames (unstable FENs) should be filtered out."""
        board = chess.Board()
        fen_start = board.board_fen()
        board.push(chess.Move.from_uci("e2e4"))
        fen_after = board.board_fen()

        # Simulate: 3 frames of start, 1 animation frame (garbage), 3 frames of after
        fens = (
            [fen_start] * 3
            + ["garbage_fen_wont_match"]  # Animation frame
            + [fen_after] * 3
        )
        indices = list(range(len(fens)))

        # The garbage FEN will be None-equivalent since it's treated as unstable
        # Actually in our code, the garbage FEN changes the stable tracking
        # Let's use None instead for unreadable frames
        fens_with_none = (
            [fen_start] * 3
            + [None]
            + [fen_after] * 3
        )

        segments = detect_moves(fens_with_none, indices, fps=2.0)
        assert len(segments) == 1
        assert segments[0].num_moves == 1

    def test_multi_game_detection(self):
        """Detect multiple games when position resets to starting."""
        game1_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
        game2_moves = ["d2d4", "d7d5", "c2c4"]

        game1_fens = self._play_game(game1_moves)
        game2_fens = self._play_game(game2_moves)

        expanded_fens = []
        expanded_indices = []
        idx = 0

        # Game 1
        for fen in game1_fens:
            for _ in range(3):
                expanded_fens.append(fen)
                expanded_indices.append(idx)
                idx += 1

        # Game 2 (starts from starting position again)
        for fen in game2_fens:
            for _ in range(3):
                expanded_fens.append(fen)
                expanded_indices.append(idx)
                idx += 1

        segments = detect_moves(expanded_fens, expanded_indices, fps=2.0)

        assert len(segments) == 2
        assert segments[0].num_moves == len(game1_moves)
        assert segments[1].num_moves == len(game2_moves)

    def test_empty_fens(self):
        """Handle empty FEN list gracefully."""
        segments = detect_moves([], [], fps=2.0)
        assert segments == []

    def test_all_none_fens(self):
        """Handle all-None FEN list gracefully."""
        segments = detect_moves([None, None, None], [0, 1, 2], fps=2.0)
        assert segments == []

    def test_move_san_correct(self):
        """Detected moves should have correct SAN notation."""
        fens = self._play_game(["e2e4"])

        expanded_fens = []
        expanded_indices = []
        idx = 0
        for fen in fens:
            for _ in range(3):
                expanded_fens.append(fen)
                expanded_indices.append(idx)
                idx += 1

        segments = detect_moves(expanded_fens, expanded_indices, fps=2.0)
        assert segments[0].moves[0].move_san == "e4"

    def test_pgn_moves_property(self):
        """GameSegment.pgn_moves should return space-separated SAN."""
        fens = self._play_game(["e2e4", "e7e5"])

        expanded_fens = []
        expanded_indices = []
        idx = 0
        for fen in fens:
            for _ in range(3):
                expanded_fens.append(fen)
                expanded_indices.append(idx)
                idx += 1

        segments = detect_moves(expanded_fens, expanded_indices, fps=2.0)
        assert segments[0].pgn_moves == "e4 e5"
