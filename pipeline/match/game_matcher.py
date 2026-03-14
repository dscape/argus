"""Match games to YouTube videos using multi-signal scoring."""

import json
import logging

from pipeline.db.connection import get_conn
from pipeline.match.scoring import (
    MatchSignals,
    compute_player_score,
    compute_date_score,
    compute_round_score,
    compute_result_score,
    compute_confidence,
)
from pipeline.match.pgn_verifier import verify_pgn

logger = logging.getLogger(__name__)


def match_video_to_games(
    video_id: str,
    white_fide_id: int | None,
    black_fide_id: int | None,
    extracted_year: int | None,
    extracted_event: str | None,
    extracted_round: str | None,
    extracted_result: str | None,
    video_pgn: str | None,
    published_at=None,
    min_confidence: float = 60.0,
) -> list[dict]:
    """Find matching games for a video.

    Returns list of matches sorted by confidence, above min_confidence.
    """
    if not white_fide_id and not black_fide_id:
        return []

    # Query candidate games
    candidates = _query_candidates(
        white_fide_id, black_fide_id, extracted_year
    )

    if not candidates:
        return []

    matches = []
    for game in candidates:
        signals = MatchSignals()

        # Player match
        signals.player_match, signals.color_correct = compute_player_score(
            game["white_fide_id"], game["black_fide_id"],
            white_fide_id, black_fide_id,
        )

        # Event similarity (use pg_trgm if both events present)
        if extracted_event and game["event"]:
            signals.event_similarity = _event_similarity(
                extracted_event, game["event"]
            )

        # Date proximity
        video_date = published_at.date() if published_at else None
        signals.date_proximity = compute_date_score(
            game["date"], extracted_year, video_date
        )

        # Round match
        signals.round_match = compute_round_score(
            game["round"], extracted_round
        )

        # Result match
        signals.result_match = compute_result_score(
            game["result"], extracted_result
        )

        # PGN verification
        if video_pgn and game["pgn_moves"]:
            signals.pgn_verify = verify_pgn(game["pgn_moves"], video_pgn)

        confidence = compute_confidence(signals)

        if confidence >= min_confidence:
            matches.append({
                "game_id": game["id"],
                "confidence": confidence,
                "signals": signals.to_dict(),
            })

    # Sort by confidence descending, return top matches
    matches.sort(key=lambda m: m["confidence"], reverse=True)
    return matches


def _query_candidates(
    white_fide_id: int | None,
    black_fide_id: int | None,
    year: int | None,
) -> list[dict]:
    """Query candidate games from the database."""
    conditions = []
    params = []

    if white_fide_id and black_fide_id:
        conditions.append(
            """
            ((white_fide_id = %s AND black_fide_id = %s)
             OR (white_fide_id = %s AND black_fide_id = %s))
            """
        )
        params.extend([white_fide_id, black_fide_id,
                       black_fide_id, white_fide_id])
    elif white_fide_id:
        conditions.append(
            "(white_fide_id = %s OR black_fide_id = %s)"
        )
        params.extend([white_fide_id, white_fide_id])
    elif black_fide_id:
        conditions.append(
            "(white_fide_id = %s OR black_fide_id = %s)"
        )
        params.extend([black_fide_id, black_fide_id])

    if year:
        conditions.append("EXTRACT(YEAR FROM date) = %s")
        params.append(year)

    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, white_name, black_name,
                       white_fide_id, black_fide_id,
                       event, site, date, round, result,
                       pgn_moves
                FROM games
                WHERE {where}
                LIMIT 100
                """,
                params,
            )
            columns = [
                "id", "white_name", "black_name",
                "white_fide_id", "black_fide_id",
                "event", "site", "date", "round", "result",
                "pgn_moves",
            ]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def _event_similarity(event1: str, event2: str) -> float:
    """Compute pg_trgm similarity between two event names."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT similarity(%s, %s)",
                (event1, event2),
            )
            return cur.fetchone()[0]
