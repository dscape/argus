"""Filter video titles for likely OTB chess footage.

A lightweight pre-filter that avoids expensive frame sampling on videos
unlikely to contain usable training data. Positive signals include "vs",
tournament names, and player-name-like patterns. Negative signals exclude
lessons, puzzles, and non-game content.
"""

import re

# ── Positive patterns ──────────────────────────────────────────────────

# "Player A vs Player B", "Player A v. Player B", "Player A versus Player B"
_VS_PATTERN = re.compile(
    r"\b(?:versus|vs\.?|v\.)\s",
    re.IGNORECASE,
)

# Two capitalized multi-word names separated by vs/v.
# e.g. "Magnus Carlsen vs Ian Nepomniachtchi"
_PLAYER_VS_PATTERN = re.compile(
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:vs?\.?|versus)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
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
    r"\b(?:round|game|match|classical|rapid|blitz|tournament|championship)\b",
    re.IGNORECASE,
)

# ── Negative patterns ──────────────────────────────────────────────────

_NEGATIVE_KEYWORDS = re.compile(
    r"\b(?:puzzle|lesson|tutorial|opening theory|endgame study|"
    r"stream highlights|subscriber game|top\s*\d+|best of|"
    r"how to|beginner|learn chess|chess tips|rating climb|"
    r"guess the elo|tier list)\b",
    re.IGNORECASE,
)


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

    score = 0.0

    # "vs" pattern — strong signal
    if _VS_PATTERN.search(title):
        score += 0.4

    # Full "Player A vs Player B" pattern — very strong
    if _PLAYER_VS_PATTERN.search(title):
        score += 0.3

    # Tournament name — strong signal
    if _TOURNAMENT_PATTERN.search(title):
        score += 0.3

    # Context keywords — mild boost
    context_matches = len(_CONTEXT_KEYWORDS.findall(title))
    score += min(context_matches * 0.1, 0.2)

    # Threshold: need at least 0.3 to be a candidate
    is_candidate = score >= 0.3
    confidence = min(score, 1.0)

    return is_candidate, confidence
