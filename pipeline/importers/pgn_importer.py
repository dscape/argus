"""Import PGN games from pgns.zip into the games table."""

import io
import os
import re
import zipfile
from datetime import date

import chess.pgn
from tqdm import tqdm

from pipeline.db.connection import get_conn

PGNS_ZIP = os.path.join("data", "chess", "pgns.zip")

# Batch size for COPY inserts
BATCH_SIZE = 5000


def _parse_date(date_str: str) -> date | None:
    """Parse PGN date format YYYY.MM.DD, handling partial dates."""
    if not date_str or date_str == "????.??.??":
        return None
    # Replace unknown parts
    parts = date_str.split(".")
    try:
        year = int(parts[0]) if parts[0] != "????" else None
        month = int(parts[1]) if len(parts) > 1 and parts[1] != "??" else 1
        day = int(parts[2]) if len(parts) > 2 and parts[2] != "??" else 1
        if year is None:
            return None
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def _parse_elo(elo_str: str) -> int | None:
    """Parse Elo rating string to int."""
    if not elo_str or elo_str == "0":
        return None
    try:
        return int(elo_str)
    except ValueError:
        return None


def _parse_fide_id(fide_str: str) -> int | None:
    """Parse FIDE ID string to int."""
    if not fide_str or fide_str == "0":
        return None
    try:
        return int(fide_str)
    except ValueError:
        return None


def _game_to_movetext(game: chess.pgn.Game) -> str:
    """Extract movetext (SAN moves) from a parsed game."""
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return game.accept(exporter).strip()


def import_pgns(zip_path: str = PGNS_ZIP, limit: int | None = None):
    """Import PGN games from a zip file into the database.

    Args:
        zip_path: Path to the pgns.zip file.
        limit: Optional limit on number of games to import (for testing).
    """
    with zipfile.ZipFile(zip_path) as z:
        pgn_files = sorted(
            [n for n in z.namelist() if n.endswith(".pgn") and not n.startswith("__MACOSX")]
        )
        print(f"Found {len(pgn_files)} PGN files in {zip_path}")

        total_imported = 0
        batch = []

        with get_conn() as conn:
            with conn.cursor() as cur:
                for pgn_file in tqdm(pgn_files, desc="PGN files"):
                    with z.open(pgn_file) as f:
                        text = f.read().decode("latin-1")

                    pgn_stream = io.StringIO(text)
                    # Extract just the filename without directory
                    twic_name = os.path.basename(pgn_file)

                    while True:
                        game = chess.pgn.read_game(pgn_stream)
                        if game is None:
                            break

                        headers = game.headers
                        movetext = _game_to_movetext(game)
                        if not movetext or movetext == "*":
                            continue

                        row = (
                            headers.get("White", ""),
                            headers.get("Black", ""),
                            _parse_fide_id(headers.get("WhiteFideId", "")),
                            _parse_fide_id(headers.get("BlackFideId", "")),
                            headers.get("Event", ""),
                            headers.get("Site", ""),
                            _parse_date(headers.get("Date", "")),
                            headers.get("Round", ""),
                            headers.get("Result", ""),
                            _parse_elo(headers.get("WhiteElo", "")),
                            _parse_elo(headers.get("BlackElo", "")),
                            headers.get("ECO", ""),
                            movetext,
                            twic_name,
                        )
                        batch.append(row)

                        if len(batch) >= BATCH_SIZE:
                            _insert_batch(cur, batch)
                            conn.commit()
                            total_imported += len(batch)
                            batch = []

                            if limit and total_imported >= limit:
                                print(f"Reached limit of {limit} games")
                                return

                # Final batch
                if batch:
                    _insert_batch(cur, batch)
                    conn.commit()
                    total_imported += len(batch)

        print(f"Imported {total_imported} games total")


def _insert_batch(cur, batch: list[tuple]):
    """Insert a batch of games using executemany."""
    cur.executemany(
        """
        INSERT INTO games
            (white_name, black_name, white_fide_id, black_fide_id,
             event, site, date, round, result,
             white_elo, black_elo, eco, pgn_moves, twic_file)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        batch,
    )


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    import_pgns(limit=limit)
