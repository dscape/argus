"""Parse chess player names, events, and metadata from YouTube video titles."""

import re
from dataclasses import dataclass, field

# Chess title prefixes to strip
TITLE_PREFIXES = re.compile(
    r"\b(GM|IM|FM|WGM|WIM|WFM|CM|WCM|NM|WNM)\b\.?\s*", re.IGNORECASE
)

# "vs" separator variants
VS_PATTERN = re.compile(
    r"\s+(?:vs\.?|versus|v\.?|against)\s+", re.IGNORECASE
)

# Year pattern (4 digits, 1900-2099)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Round patterns
ROUND_PATTERN = re.compile(
    r"\b(?:R(?:ound|d)?\.?\s*(\d+(?:\.\d+)?))\b", re.IGNORECASE
)

# Result patterns
RESULT_PATTERN = re.compile(
    r"\b(1-0|0-1|1/2-1/2|½-½|draw)\b", re.IGNORECASE
)

# Common separators between player names and event info
SEPARATOR_PATTERN = re.compile(r"\s*[|•·—–-]\s*")


@dataclass
class TitleExtraction:
    """Structured data extracted from a video title."""
    white: str | None = None
    black: str | None = None
    event: str | None = None
    year: int | None = None
    round: str | None = None
    result: str | None = None
    confidence: float = 0.0


def _strip_titles(name: str) -> str:
    """Remove chess title prefixes (GM, IM, etc.) from a player name."""
    return TITLE_PREFIXES.sub("", name).strip()


def _clean_name(name: str) -> str:
    """Clean up a player name string."""
    name = _strip_titles(name)
    # Remove parenthetical info like "(2800)"
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    # Remove rating numbers
    name = re.sub(r"\s+\d{3,4}\s*$", "", name)
    return name.strip()


def parse_title(title: str) -> TitleExtraction:
    """Extract player names, event, year, round, and result from a video title.

    Handles common patterns:
    - "Player A vs Player B | Event Name Round X"
    - "Player A vs Player B - Event YYYY"
    - "Player A - Player B, Event"
    - "Event: Player A vs Player B"
    """
    result = TitleExtraction()

    # Extract year first (remove from further parsing)
    year_match = YEAR_PATTERN.search(title)
    if year_match:
        result.year = int(year_match.group(1))

    # Extract round
    round_match = ROUND_PATTERN.search(title)
    if round_match:
        result.round = round_match.group(1)

    # Extract result
    result_match = RESULT_PATTERN.search(title)
    if result_match:
        r = result_match.group(1).lower()
        if r == "draw" or r == "½-½":
            result.result = "1/2-1/2"
        else:
            result.result = result_match.group(1)

    # Remove result from title before separator splitting
    # (prevents "1-0" from being split on the dash)
    working_title = RESULT_PATTERN.sub("", title).strip()

    # Try to find "vs" pattern
    vs_match = VS_PATTERN.search(working_title)
    if vs_match:
        before_vs = working_title[:vs_match.start()]
        after_vs = working_title[vs_match.end():]

        # Split before_vs by separators to get player name and possible prefix event
        before_parts = SEPARATOR_PATTERN.split(before_vs)
        if len(before_parts) > 1:
            # Last part before separator is likely player name, earlier parts are event
            result.white = _clean_name(before_parts[-1])
            result.event = " ".join(before_parts[:-1]).strip()
        else:
            result.white = _clean_name(before_vs)

        # Split after_vs by separators to get player name and possible suffix event
        after_parts = SEPARATOR_PATTERN.split(after_vs)
        result.black = _clean_name(after_parts[0])

        # Remaining parts after black player are likely event info
        if len(after_parts) > 1 and not result.event:
            event_parts = " ".join(after_parts[1:]).strip()
            # Remove year and round from event string
            event_parts = YEAR_PATTERN.sub("", event_parts)
            event_parts = ROUND_PATTERN.sub("", event_parts)
            event_parts = RESULT_PATTERN.sub("", event_parts)
            # Collapse multiple spaces and strip separators
            event_parts = re.sub(r"\s+", " ", event_parts).strip(" |•·—–-,")
            if event_parts:
                result.event = event_parts

        # Confidence based on what we found
        result.confidence = 0.0
        if result.white and result.black:
            result.confidence = 0.7
            if result.year:
                result.confidence += 0.1
            if result.event:
                result.confidence += 0.1
            if result.round:
                result.confidence += 0.05
            if result.result:
                result.confidence += 0.05
    else:
        # No "vs" found — try "Player1 - Player2" with dash separator
        # Only if there's a clear separator and both sides look like names
        parts = SEPARATOR_PATTERN.split(working_title, maxsplit=2)
        if len(parts) >= 2:
            name1 = _clean_name(parts[0])
            name2 = _clean_name(parts[1])
            # Heuristic: names are 2-40 chars, don't contain chess keywords
            chess_keywords = {"chess", "game", "match", "tournament", "analysis",
                            "opening", "endgame", "middlegame", "brilliant", "blunder"}
            if (2 < len(name1) < 40 and 2 < len(name2) < 40
                    and not any(kw in name1.lower() for kw in chess_keywords)
                    and not any(kw in name2.lower() for kw in chess_keywords)):
                result.white = name1
                result.black = name2
                result.confidence = 0.4  # Lower confidence without "vs"
                if len(parts) > 2:
                    event_text = parts[2].strip()
                    event_text = YEAR_PATTERN.sub("", event_text)
                    event_text = ROUND_PATTERN.sub("", event_text)
                    event_text = event_text.strip(" |•·—–-,")
                    if event_text:
                        result.event = event_text
                        result.confidence += 0.1

    return result


def parse_chapter_title(chapter_title: str) -> TitleExtraction:
    """Parse a chapter title (typically shorter, simpler format).

    Chapter titles are often just "Player A vs Player B" without event info.
    """
    return parse_title(chapter_title)
