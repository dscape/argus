"""Orchestrate metadata extraction from YouTube video titles and descriptions."""

import logging

from tqdm import tqdm

from pipeline.db.connection import get_conn
from pipeline.extract.title_parser import parse_title, parse_chapter_title
from pipeline.extract.description_parser import parse_chapters, extract_pgn
from pipeline.extract.player_normalizer import PlayerNormalizer

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000


def extract_all(use_claude: bool = False):
    """Run metadata extraction on all unprocessed videos.

    Steps:
    1. Parse titles with regex
    2. Parse descriptions for chapters and PGN
    3. Normalize player names to FIDE IDs
    4. (Optional) Run Claude on low-confidence titles
    5. Re-normalize Claude results
    """
    normalizer = PlayerNormalizer()

    # Phase 1: Regex extraction + description parsing
    _extract_regex_pass(normalizer)

    # Phase 2: Claude fallback for low-confidence titles
    if use_claude:
        from pipeline.extract.claude_extractor import ClaudeExtractor
        extractor = ClaudeExtractor()
        extractor.extract_and_store()
        # Re-normalize Claude-extracted names
        _normalize_claude_results(normalizer)

    _print_stats()


def _extract_regex_pass(normalizer: PlayerNormalizer):
    """Run regex-based extraction on all unprocessed videos."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM youtube_videos
                WHERE extraction_confidence IS NULL
                """
            )
            total = cur.fetchone()[0]

    if total == 0:
        print("No unprocessed videos found.")
        return

    print(f"Processing {total} videos with regex extraction...")
    offset = 0
    processed = 0
    chapters_total = 0

    while offset < total:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT video_id, title, description
                    FROM youtube_videos
                    WHERE extraction_confidence IS NULL
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
                for video_id, title, description in tqdm(
                    videos, desc="Extracting", leave=False
                ):
                    # Parse title
                    extraction = parse_title(title)

                    # Normalize player names to FIDE IDs
                    white_fide_id = None
                    black_fide_id = None
                    if extraction.white:
                        white_fide_id, _ = normalizer.normalize(extraction.white)
                    if extraction.black:
                        black_fide_id, _ = normalizer.normalize(extraction.black)

                    # Extract PGN from description
                    pgn = None
                    has_pgn = False
                    if description:
                        pgn = extract_pgn(description)
                        has_pgn = pgn is not None

                    # Update video record
                    cur.execute(
                        """
                        UPDATE youtube_videos
                        SET extracted_white = %s,
                            extracted_black = %s,
                            white_fide_id = %s,
                            black_fide_id = %s,
                            extracted_event = %s,
                            extracted_year = %s,
                            extracted_round = %s,
                            extracted_result = %s,
                            extraction_method = 'regex',
                            extraction_confidence = %s,
                            has_pgn = %s,
                            video_pgn = %s,
                            updated_at = now()
                        WHERE video_id = %s
                        """,
                        (
                            extraction.white,
                            extraction.black,
                            white_fide_id,
                            black_fide_id,
                            extraction.event,
                            extraction.year,
                            extraction.round,
                            extraction.result,
                            extraction.confidence,
                            has_pgn,
                            pgn,
                            video_id,
                        ),
                    )

                    # Parse chapters from description
                    if description:
                        chapters = parse_chapters(description)
                        for ch in chapters:
                            ch_extraction = parse_chapter_title(ch.title)
                            ch_white_fide = None
                            ch_black_fide = None
                            if ch_extraction.white:
                                ch_white_fide, _ = normalizer.normalize(
                                    ch_extraction.white
                                )
                            if ch_extraction.black:
                                ch_black_fide, _ = normalizer.normalize(
                                    ch_extraction.black
                                )

                            cur.execute(
                                """
                                INSERT INTO video_chapters
                                    (video_id, timestamp_seconds, chapter_title,
                                     extracted_white, extracted_black,
                                     white_fide_id, black_fide_id,
                                     extraction_confidence)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (video_id, timestamp_seconds) DO NOTHING
                                """,
                                (
                                    video_id,
                                    ch.timestamp_seconds,
                                    ch.title,
                                    ch_extraction.white,
                                    ch_extraction.black,
                                    ch_white_fide,
                                    ch_black_fide,
                                    ch_extraction.confidence,
                                ),
                            )
                            chapters_total += 1

                    processed += 1

                conn.commit()

        offset += BATCH_SIZE

    print(f"Processed {processed} videos, extracted {chapters_total} chapters")


def _normalize_claude_results(normalizer: PlayerNormalizer):
    """Normalize player names from Claude extraction to FIDE IDs."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id, extracted_white, extracted_black
                FROM youtube_videos
                WHERE extraction_method = 'claude'
                  AND (white_fide_id IS NULL OR black_fide_id IS NULL)
                  AND (extracted_white IS NOT NULL OR extracted_black IS NOT NULL)
                """
            )
            videos = cur.fetchall()

    if not videos:
        return

    print(f"Normalizing {len(videos)} Claude-extracted player names...")
    with get_conn() as conn:
        with conn.cursor() as cur:
            for video_id, white, black in videos:
                white_fide = None
                black_fide = None
                if white:
                    white_fide, _ = normalizer.normalize(white)
                if black:
                    black_fide, _ = normalizer.normalize(black)

                if white_fide or black_fide:
                    cur.execute(
                        """
                        UPDATE youtube_videos
                        SET white_fide_id = COALESCE(%s, white_fide_id),
                            black_fide_id = COALESCE(%s, black_fide_id),
                            updated_at = now()
                        WHERE video_id = %s
                        """,
                        (white_fide, black_fide, video_id),
                    )
            conn.commit()


def _print_stats():
    """Print extraction statistics."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM youtube_videos")
            total = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM youtube_videos WHERE extraction_confidence > 0"
            )
            extracted = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM youtube_videos WHERE white_fide_id IS NOT NULL"
            )
            with_fide = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM youtube_videos WHERE has_pgn = true"
            )
            with_pgn = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM video_chapters")
            chapters = cur.fetchone()[0]

    print(f"\nExtraction stats:")
    print(f"  Total videos:           {total}")
    print(f"  With extraction:        {extracted}")
    print(f"  With FIDE IDs:          {with_fide}")
    print(f"  With PGN:               {with_pgn}")
    print(f"  Chapters extracted:     {chapters}")
