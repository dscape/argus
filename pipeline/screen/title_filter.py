"""Filter video titles for likely OTB chess footage.

A lightweight pre-filter that avoids expensive frame sampling on videos
unlikely to contain usable training data. Positive signals include "vs",
tournament names, and player-name-like patterns. Negative signals exclude
lessons, puzzles, clickbait, reaction content, and non-game content.
"""

import re

# ── Positive patterns ──────────────────────────────────────────────────

# "Player A vs Player B", "Player A v. Player B", "Player A versus Player B"
# Also match dash/|| separator: "Player A - Player B", "Player A || Tournament"
_VS_PATTERN = re.compile(
    r"\b(?:versus|vs\.?|v\.)\s|[A-Z][a-z]+\s+[-–—]\s+[A-Z][a-z]+|[A-Z][a-z]+\s+\|\|\s+",
    re.IGNORECASE,
)

# Two capitalized multi-word names separated by vs/v./dash
# e.g. "Magnus Carlsen vs Ian Nepomniachtchi" or "Carlsen - Nepomniachtchi"
_PLAYER_VS_PATTERN = re.compile(
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:vs?\.?|versus|[-–—])\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
)

# Known tournament / event names (case-insensitive fragments)
_TOURNAMENT_KEYWORDS = [
    "candidates", "sinquefield", "tata steel", "world championship",
    "olympiad", "grand prix", "grand chess tour", "norway chess",
    "wijk aan zee", "world cup", "world rapid", "world blitz",
    "fide", "us championship", "european championship",
    "british championship", "bundesliga", "champions chess tour",
    "fischer random", "chess960", "freestyle chess",
    "grand swiss", "isle of man", "grenke", "shamkir",
    "superbet", "chessable masters",
]

_TOURNAMENT_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in _TOURNAMENT_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Chess context words (boost when combined with other signals)
_CONTEXT_KEYWORDS = re.compile(
    r"\b(?:round\s*\d+|game\s*\d+|round|game|match|classical|rapid|blitz|tournament|championship)\b",
    re.IGNORECASE,
)

# ── Negative patterns ──────────────────────────────────────────────────

_NEGATIVE_KEYWORDS = re.compile(
    r"\b(?:puzzle|lesson|tutorial|opening theory|endgame study|"
    r"stream highlights|subscriber game|top\s*\d+|best of|"
    r"how to|beginner|learn chess|chess tips|rating climb|"
    r"guess the elo|tier list|shorts|reaction|meme|funny|"
    r"compilation|drama|lichess|chess\.com|bullet|arena|"
    r"opening trap|trick|hack|speedrun|"
    # Sensationalized / clickbait
    r"destroys|insane|incredible|unbelievable|impossible|"
    # Casual / meme
    r"rizz|cursed|"
    # First-person
    r"i got|my moves|join me|i played|i tried|"
    # Clickbait phrases
    r"you won't believe|you wont believe|wait for it|do you see it|"
    # Player reaction
    r"slams table|pushes camera|"
    # Educational
    r"secret rule|secret move|secret trick)\b",
    re.IGNORECASE,
)

# Excessive punctuation: 3+ exclamation or question marks
_EXCESSIVE_PUNCTUATION = re.compile(r"[!]{3,}|[?]{3,}")


def _is_mostly_caps(title: str) -> bool:
    """Return True if >50% of alphabetic characters are uppercase and title has 5+ words."""
    words = title.split()
    if len(words) < 5:
        return False
    alpha = [c for c in title if c.isalpha()]
    if not alpha:
        return False
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    return upper_ratio > 0.5


def score_title(title: str) -> tuple[bool, float]:
    """Score a video title for OTB chess footage likelihood.

    Returns:
        (is_candidate, confidence) where is_candidate is True if the title
        passes the filter, and confidence is a 0-1 score.
    """
    if not title or not title.strip():
        return False, 0.0

    # Negative filter — reject immediately
    if _NEGATIVE_KEYWORDS.search(title):
        return False, 0.0

    # ALL CAPS rejection — sensationalized titles
    if _is_mostly_caps(title):
        return False, 0.0

    # Excessive punctuation rejection
    if _EXCESSIVE_PUNCTUATION.search(title):
        return False, 0.0

    score = 0.0

    has_tournament = bool(_TOURNAMENT_PATTERN.search(title))
    has_player_vs = bool(_PLAYER_VS_PATTERN.search(title))

    # Question without tournament/player context — reject
    if title.rstrip().endswith("?") and not has_tournament and not has_player_vs:
        return False, 0.0

    # "vs" pattern — strong signal
    if _VS_PATTERN.search(title):
        score += 0.4

    # Full "Player A vs Player B" pattern — very strong
    if has_player_vs:
        score += 0.3

    # Tournament name — strong signal
    if has_tournament:
        score += 0.3

    # Context keywords — mild boost
    context_matches = len(_CONTEXT_KEYWORDS.findall(title))
    score += min(context_matches * 0.1, 0.2)

    # Year pattern — mild boost when combined with other signals
    if score > 0 and re.search(r"\b20[0-2]\d\b", title):
        score += 0.1

    # Threshold: need at least 0.15 to be a candidate
    is_candidate = score >= 0.15
    confidence = min(score, 1.0)

    return is_candidate, confidence
