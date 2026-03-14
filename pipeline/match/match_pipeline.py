"""Orchestrate game-video matching across all videos and chapters."""

import json
import logging
from datetime import datetime

from tqdm import tqdm

from pipeline.db.connection import get_conn
from pipeline.match.game_matcher import match_video_to_games

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def match_all(min_confidence: float = 60.0):
    """Run matching for all videos with resolved FIDE IDs.

    Matches at two levels:
    1. Video-level: for single-game videos
    2. Chapter-level: for multi-board streams
    """
    video_matches = _match_videos(min_confidence)
    chapter_matches = _match_chapters(min_confidence)
    _print_stats()
    return video_matches + chapter_matches


def _match_videos(min_confidence: float) -> int:
    """Match videos to games at the video level."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM youtube_videos v
                WHERE v.white_fide_id IS NOT NULL
                  AND v.black_fide_id IS NOT NULL
                  AND v.extraction_confidence > 0.5
                  AND NOT EXISTS (
                      SELECT 1 FROM game_video_links gvl
                      WHERE gvl.video_id = v.video_id
                  )
                """
            )
            total = cur.fetchone()[0]

    if total == 0:
        print("No videos to match at video level.")
        return 0

    print(f"Matching {total} videos to games...")
    offset = 0
    total_matches = 0

    while offset < total:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT video_id, white_fide_id, black_fide_id,
                           extracted_year, extracted_event,
                           extracted_round, extracted_result,
                           video_pgn, published_at
                    FROM youtube_videos
                    WHERE white_fide_id IS NOT NULL
                      AND black_fide_id IS NOT NULL
                      AND extraction_confidence > 0.5
                      AND NOT EXISTS (
                          SELECT 1 FROM game_video_links gvl
                          WHERE gvl.video_id = youtube_videos.video_id
                      )
                    ORDER BY video_id
                    LIMIT %s OFFSET %s
                    """,
                    (BATCH_SIZE, offset),
                )
                videos = cur.fetchall()

        if not videos:
            break

        with get_conn() as conn:
            with conn.cursor() as cur:
                for row in tqdm(videos, desc="Video matching", leave=False):
                    (video_id, white_fide, black_fide, year, event,
                     round_num, result, video_pgn, published_at) = row

                    matches = match_video_to_games(
                        video_id=video_id,
                        white_fide_id=white_fide,
                        black_fide_id=black_fide,
                        extracted_year=year,
                        extracted_event=event,
                        extracted_round=round_num,
                        extracted_result=result,
                        video_pgn=video_pgn,
                        published_at=published_at,
                        min_confidence=min_confidence,
                    )

                    for match in matches:
                        cur.execute(
                            """
                            INSERT INTO game_video_links
                                (game_id, video_id, match_confidence, match_signals)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (game_id, video_id) DO UPDATE SET
                                match_confidence = GREATEST(
                                    game_video_links.match_confidence,
                                    EXCLUDED.match_confidence
                                ),
                                match_signals = EXCLUDED.match_signals
                            """,
                            (
                                match["game_id"],
                                video_id,
                                match["confidence"],
                                json.dumps(match["signals"]),
                            ),
                        )
                        total_matches += 1

                conn.commit()

        offset += BATCH_SIZE

    print(f"Created {total_matches} video-level matches")
    return total_matches


def _match_chapters(min_confidence: float) -> int:
    """Match chapters (from multi-board streams) to games."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM video_chapters vc
                WHERE vc.white_fide_id IS NOT NULL
                  AND vc.black_fide_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM game_video_links gvl
                      WHERE gvl.video_id = vc.video_id
                        AND gvl.chapter_id = vc.id
                  )
                """
            )
            total = cur.fetchone()[0]

    if total == 0:
        print("No chapters to match.")
        return 0

    print(f"Matching {total} chapters to games...")
    offset = 0
    total_matches = 0

    while offset < total:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT vc.id, vc.video_id, vc.timestamp_seconds,
                           vc.white_fide_id, vc.black_fide_id,
                           v.extracted_year, v.extracted_event, v.published_at
                    FROM video_chapters vc
                    JOIN youtube_videos v ON v.video_id = vc.video_id
                    WHERE vc.white_fide_id IS NOT NULL
                      AND vc.black_fide_id IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM game_video_links gvl
                          WHERE gvl.video_id = vc.video_id
                            AND gvl.chapter_id = vc.id
                      )
                    ORDER BY vc.id
                    LIMIT %s OFFSET %s
                    """,
                    (BATCH_SIZE, offset),
                )
                chapters = cur.fetchall()

        if not chapters:
            break

        with get_conn() as conn:
            with conn.cursor() as cur:
                for row in tqdm(chapters, desc="Chapter matching", leave=False):
                    (ch_id, video_id, timestamp, white_fide, black_fide,
                     year, event, published_at) = row

                    matches = match_video_to_games(
                        video_id=video_id,
                        white_fide_id=white_fide,
                        black_fide_id=black_fide,
                        extracted_year=year,
                        extracted_event=event,
                        extracted_round=None,
                        extracted_result=None,
                        video_pgn=None,
                        published_at=published_at,
                        min_confidence=min_confidence,
                    )

                    for match in matches:
                        cur.execute(
                            """
                            INSERT INTO game_video_links
                                (game_id, video_id, chapter_id,
                                 timestamp_seconds, match_confidence,
                                 match_signals)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (game_id, video_id) DO UPDATE SET
                                match_confidence = GREATEST(
                                    game_video_links.match_confidence,
                                    EXCLUDED.match_confidence
                                ),
                                match_signals = EXCLUDED.match_signals,
                                chapter_id = EXCLUDED.chapter_id,
                                timestamp_seconds = EXCLUDED.timestamp_seconds
                            """,
                            (
                                match["game_id"],
                                video_id,
                                ch_id,
                                timestamp,
                                match["confidence"],
                                json.dumps(match["signals"]),
                            ),
                        )
                        total_matches += 1

                conn.commit()

        offset += BATCH_SIZE

    print(f"Created {total_matches} chapter-level matches")
    return total_matches


def _print_stats():
    """Print matching statistics."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM game_video_links")
            total = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM game_video_links WHERE match_confidence >= 80"
            )
            high_conf = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM game_video_links WHERE chapter_id IS NOT NULL"
            )
            chapter_links = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM game_video_links WHERE verified = true"
            )
            verified = cur.fetchone()[0]

            cur.execute(
                "SELECT AVG(match_confidence) FROM game_video_links"
            )
            avg_conf = cur.fetchone()[0] or 0

    print(f"\nMatching stats:")
    print(f"  Total links:           {total}")
    print(f"  High confidence (>=80): {high_conf}")
    print(f"  Chapter-level links:   {chapter_links}")
    print(f"  PGN-verified:          {verified}")
    print(f"  Avg confidence:        {avg_conf:.1f}")
