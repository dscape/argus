"""Regression tests for FEN validation and grid-to-FEN conversion.

Covers _is_valid_fen_placement and _build_fen_from_class_grid from
overlay_test_service — ensures invalid extractions (missing kings)
are correctly flagged so the UI can surface them for review.
"""

from api.services.evaluate.overlay_test_service import (
    _build_fen_from_class_grid,
    _is_valid_fen_placement,
)


class TestIsValidFenPlacement:
    """_is_valid_fen_placement must reject FENs missing either king."""

    def test_starting_position_valid(self) -> None:
        assert _is_valid_fen_placement("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    def test_clip42_no_kings_invalid(self) -> None:
        """Regression: clip 42 (Unu6antTBGs) produced a FEN with no kings."""
        assert not _is_valid_fen_placement("8/7r/5r1p/5rnq/6Bp/6Pr/8/8")

    def test_missing_white_king_invalid(self) -> None:
        assert not _is_valid_fen_placement("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR")

    def test_missing_black_king_invalid(self) -> None:
        assert not _is_valid_fen_placement("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    def test_empty_board_invalid(self) -> None:
        assert not _is_valid_fen_placement("8/8/8/8/8/8/8/8")

    def test_endgame_with_both_kings_valid(self) -> None:
        assert _is_valid_fen_placement("8/8/4k3/8/8/4K3/8/8")

    def test_full_fen_string_valid(self) -> None:
        """Accepts a full FEN (with side-to-move, castling, etc.)."""
        assert _is_valid_fen_placement(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )

    def test_garbage_string_invalid(self) -> None:
        assert not _is_valid_fen_placement("not-a-fen")


class TestBuildFenFromClassGrid:
    """_build_fen_from_class_grid must produce correct FEN from class indices."""

    # Class indices: 0=empty, 1=P,2=N,3=B,4=R,5=Q,6=K,7=p,8=n,9=b,10=r,11=q,12=k

    def test_empty_board(self) -> None:
        grid = [[0] * 8 for _ in range(8)]
        assert _build_fen_from_class_grid(grid) == "8/8/8/8/8/8/8/8"

    def test_starting_position(self) -> None:
        grid = [
            [10, 8, 9, 11, 12, 9, 8, 10],  # r n b q k b n r
            [7, 7, 7, 7, 7, 7, 7, 7],  # p p p p p p p p
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],  # P P P P P P P P
            [4, 2, 3, 5, 6, 3, 2, 4],  # R N B Q K B N R
        ]
        fen = _build_fen_from_class_grid(grid)
        assert fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    def test_no_king_grid_flagged_invalid(self) -> None:
        """Round-trip: a grid with no kings produces an invalid FEN."""
        grid = [[0] * 8 for _ in range(8)]
        grid[0][0] = 10  # r
        grid[7][0] = 4  # R
        fen = _build_fen_from_class_grid(grid)
        assert not _is_valid_fen_placement(fen)

    def test_mixed_pieces_and_empties(self) -> None:
        grid = [[0] * 8 for _ in range(8)]
        grid[0][4] = 12  # black king on e8
        grid[7][4] = 6  # white king on e1
        grid[3][3] = 1  # white pawn on d5
        fen = _build_fen_from_class_grid(grid)
        assert fen == "4k3/8/8/3P4/8/8/8/4K3"
        assert _is_valid_fen_placement(fen)
