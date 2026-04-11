from __future__ import annotations

from argus.chess.board_state import fen_to_square_targets


def test_fen_to_square_targets_maps_piece_symbols_to_indices() -> None:
    targets = fen_to_square_targets("8/8/8/8/8/8/8/K6k w - - 0 1")

    assert targets.shape == (64,)
    assert targets[56].item() == 6
    assert targets[63].item() == 12


def test_fen_to_square_targets_rotates_for_flipped_board() -> None:
    targets = fen_to_square_targets("8/8/8/8/8/8/8/K6k w - - 0 1", board_flipped=True)

    assert targets[0].item() == 12
    assert targets[7].item() == 6
