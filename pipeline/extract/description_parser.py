"""Parse YouTube video descriptions for chapter timestamps and PGN data."""

import io
import re
from dataclasses import dataclass

import chess.pgn


@dataclass
class Chapter:
    """A chapter (timestamp) from a video description."""
    timestamp_seconds: int
    title: str


def parse_chapters(description: str) -> list[Chapter]:
    """Extract chapter timestamps from a video description.

    YouTube chapter format:
        0:00 Introduction
        0:45 Carlsen vs Nepomniachtchi
        1:23:10 Caruana vs Giri
    """
    chapters = []
    # Match lines starting with a timestamp
    pattern = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$", re.MULTILINE)

    for match in pattern.finditer(description):
        timestamp_str = match.group(1)
        title = match.group(2).strip()

        # Parse timestamp to seconds
        parts = timestamp_str.split(":")
        if len(parts) == 2:
            seconds = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            continue

        chapters.append(Chapter(timestamp_seconds=seconds, title=title))

    return chapters


def extract_pgn(description: str) -> str | None:
    """Extract PGN movetext from a video description.

    Looks for chess move notation patterns (e.g., "1. e4 e5 2. Nf3 Nc6...")
    and validates by attempting to parse with python-chess.
    """
    # Pattern to find PGN-like content: starts with "1." followed by a chess move
    pgn_pattern = re.compile(
        r"(1\.\s*[a-hNBRQKO][^\n]*(?:\n[^\n]*)*)",
        re.MULTILINE,
    )

    for match in pgn_pattern.finditer(description):
        candidate = match.group(1).strip()

        # Clean up the candidate
        # Remove URLs
        candidate = re.sub(r"https?://\S+", "", candidate)
        # Remove common non-PGN lines
        lines = candidate.split("\n")
        pgn_lines = []
        for line in lines:
            line = line.strip()
            # Stop at lines that don't look like PGN
            if not line:
                continue
            if re.match(r"\d+\.", line) or re.match(r"[a-hNBRQKO]", line):
                pgn_lines.append(line)
            elif any(c in line for c in ["0-1", "1-0", "1/2-1/2", "*"]):
                pgn_lines.append(line)
                break
            else:
                break

        if not pgn_lines:
            continue

        pgn_text = " ".join(pgn_lines)

        # Validate by parsing
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if game and len(list(game.mainline_moves())) >= 5:
                return pgn_text
        except Exception:
            continue

    return None


def extract_event_from_description(description: str) -> str | None:
    """Try to extract event/tournament name from the first few lines of a description."""
    lines = description.strip().split("\n")[:5]
    # Common patterns: "Event: Tata Steel 2024", "Tournament: ..."
    for line in lines:
        line = line.strip()
        match = re.match(
            r"(?:Event|Tournament|Competition|Match)\s*[:|-]\s*(.+)",
            line,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
    return None
