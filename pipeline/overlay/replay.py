"""Helpers for replaying move sequences from piece-placement-only board states."""

from __future__ import annotations

import chess


def build_replay_board(initial_board_fen: str, first_move_uci: str | None = None) -> chess.Board:
    """Build a board for replay from a stored piece-placement FEN.

    Real clips currently persist only ``board_fen()`` piece placement, not a full
    FEN with side-to-move / castling rights / en-passant state. For replay and
    legality checks we infer the side to move from the first move when possible
    and otherwise fall back to White-to-move.
    """

    board = chess.Board()
    board.set_board_fen(initial_board_fen)

    if first_move_uci is None:
        return board

    try:
        first_move = chess.Move.from_uci(first_move_uci)
    except ValueError:
        return board

    white_board = board.copy(stack=False)
    white_board.turn = chess.WHITE
    black_board = board.copy(stack=False)
    black_board.turn = chess.BLACK

    white_legal = first_move in white_board.legal_moves
    black_legal = first_move in black_board.legal_moves

    if black_legal and not white_legal:
        return black_board
    return white_board
