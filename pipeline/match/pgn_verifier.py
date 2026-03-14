"""Verify game matches by comparing PGN move sequences."""

import io
import logging

import chess
import chess.pgn

logger = logging.getLogger(__name__)


def verify_pgn(
    game_pgn_moves: str,
    video_pgn: str,
    num_moves: int = 15,
) -> float:
    """Compare the first N moves of two PGNs.

    Args:
        game_pgn_moves: Movetext from the games table (SAN, no headers).
        video_pgn: PGN from video description (may include headers).
        num_moves: Number of moves to compare.

    Returns:
        Score 0.0-1.0 based on matching prefix length.
    """
    game_moves = _parse_moves(game_pgn_moves)
    video_moves = _parse_moves(video_pgn)

    if not game_moves or not video_moves:
        return 0.0

    # Compare UCI move sequences
    compare_length = min(num_moves, len(game_moves), len(video_moves))
    if compare_length == 0:
        return 0.0

    matching = 0
    for i in range(compare_length):
        if game_moves[i] == video_moves[i]:
            matching += 1
        else:
            break

    score = matching / compare_length

    if matching >= 15:
        # 15+ consecutive matching moves is near-certain
        score = 1.0
    elif matching >= 10:
        score = max(score, 0.9)

    return score


def _parse_moves(pgn_text: str) -> list[str]:
    """Parse PGN text into a list of UCI moves."""
    if not pgn_text:
        return []

    try:
        # Try parsing as full PGN (with or without headers)
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return []
        return [move.uci() for move in game.mainline_moves()]
    except Exception:
        return []
