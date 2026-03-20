"""Tests for pipeline.overlay.overlay_reader."""

import chess
import cv2
import numpy as np
import pytest

from pipeline.overlay.overlay_reader import OverlayReader, PIECE_CLASSES, PIECE_TO_CLASS


class TestOverlayReaderWithRenderedBoards:
    """Test overlay reader against synthetically rendered boards."""

    @pytest.fixture
    def reader(self):
        return OverlayReader(board_theme="lichess_default")

    def _render_test_board(self, board: chess.Board, size: int = 512, flipped: bool = False) -> np.ndarray:
        """Render a board using the same renderer as the overlay reader templates."""
        from pipeline.overlay.overlay_reader import _render_board_to_cv2

        return _render_board_to_cv2(
            board,
            size=size,
            flipped=flipped,
            colors={"light": "#F0D9B5", "dark": "#B58863"},
        )

    def test_starting_position(self, reader):
        """Reader should correctly identify the standard starting position."""
        board = chess.Board()
        img = self._render_test_board(board)

        result = reader.read_board(img)
        assert result is not None
        assert result.board_fen() == board.board_fen()

    def test_empty_board(self, reader):
        """Reader should handle an empty board (just kings for validity)."""
        board = chess.Board(fen=None)
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))

        img = self._render_test_board(board)
        result = reader.read_board(img)

        assert result is not None
        # Should have exactly 2 pieces
        piece_count = sum(1 for sq in chess.SQUARES if result.piece_at(sq) is not None)
        assert piece_count == 2

    def test_mid_game_position(self, reader):
        """Reader should handle a mid-game position."""
        board = chess.Board()
        # Play Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
        for move in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]:
            board.push(chess.Move.from_uci(move))

        img = self._render_test_board(board)
        result = reader.read_board(img)

        assert result is not None
        assert result.board_fen() == board.board_fen()

    def test_flipped_board(self, reader):
        """Reader should handle flipped boards correctly."""
        board = chess.Board()
        img = self._render_test_board(board, flipped=True)

        result = reader.read_board(img, flipped=True)
        assert result is not None
        assert result.board_fen() == board.board_fen()

    def test_different_sizes(self, reader):
        """Reader should work at various image sizes."""
        board = chess.Board()

        for size in [256, 480, 512, 800]:
            img = self._render_test_board(board, size=size)
            result = reader.read_board(img)
            assert result is not None, f"Failed at size {size}"
            assert result.board_fen() == board.board_fen(), f"Wrong FEN at size {size}"

    def test_fen_extraction(self, reader):
        """read_fen should return correct FEN string."""
        board = chess.Board()
        img = self._render_test_board(board)

        fen = reader.read_fen(img)
        assert fen is not None
        assert fen == board.board_fen()


class TestBoardValidation:
    """Test the board validation logic."""

    @pytest.fixture
    def reader(self):
        return OverlayReader()

    def test_valid_board(self, reader):
        board = chess.Board()
        assert reader._validate_board(board) is True

    def test_no_white_king(self, reader):
        board = chess.Board(fen=None)
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        assert reader._validate_board(board) is False

    def test_no_black_king(self, reader):
        board = chess.Board(fen=None)
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        assert reader._validate_board(board) is False

    def test_pawn_on_first_rank(self, reader):
        board = chess.Board(fen=None)
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A1, chess.Piece(chess.PAWN, chess.WHITE))
        assert reader._validate_board(board) is False

    def test_pawn_on_eighth_rank(self, reader):
        board = chess.Board(fen=None)
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.A8, chess.Piece(chess.PAWN, chess.BLACK))
        assert reader._validate_board(board) is False
