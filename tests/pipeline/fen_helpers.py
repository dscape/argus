"""Shared helpers for FEN comparison tests."""

from __future__ import annotations

import chess
import pytest


def fen_to_board(fen: str) -> chess.Board:
    """Parse a piece-placement-only or full FEN string."""
    if " " not in fen:
        fen = fen + " w - - 0 1"
    return chess.Board(fen)


def compare_boards(predicted: chess.Board, expected: chess.Board, label: str) -> None:
    """Assert that two boards are identical, with a helpful diff on failure."""
    mismatches: list[str] = []
    for sq in chess.SQUARES:
        p = predicted.piece_at(sq)
        e = expected.piece_at(sq)
        if p != e:
            sq_name = chess.square_name(sq)
            p_str = p.symbol() if p else "."
            e_str = e.symbol() if e else "."
            mismatches.append(f"  {sq_name}: got {p_str}, expected {e_str}")

    if mismatches:
        pred_str = "\n".join(f"    {line}" for line in str(predicted).split("\n"))
        exp_str = "\n".join(f"    {line}" for line in str(expected).split("\n"))
        diff = "\n".join(mismatches)
        pytest.fail(
            f"{label}: {len(mismatches)} square(s) wrong\n"
            f"  Predicted:\n{pred_str}\n"
            f"  Expected:\n{exp_str}\n"
            f"  Mismatches:\n{diff}"
        )
