"""Match confidence scoring for game-video links."""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class MatchSignals:
    """Individual scoring signals for a game-video match."""
    player_match: float = 0.0       # Both FIDE IDs match
    color_correct: bool = True      # White/black assignment correct
    event_similarity: float = 0.0   # pg_trgm similarity of event names
    date_proximity: float = 0.0     # Closeness of dates
    round_match: float = 0.0        # Round number matches
    result_match: float = 0.0       # Game result matches
    pgn_verify: float = 0.0         # PGN move verification score

    def to_dict(self) -> dict:
        return {
            "player_match": self.player_match,
            "color_correct": self.color_correct,
            "event_similarity": self.event_similarity,
            "date_proximity": self.date_proximity,
            "round_match": self.round_match,
            "result_match": self.result_match,
            "pgn_verify": self.pgn_verify,
        }


# Default scoring weights
DEFAULT_WEIGHTS = {
    "player_match": 0.35,
    "event_similarity": 0.20,
    "date_proximity": 0.15,
    "round_match": 0.10,
    "result_match": 0.05,
    "pgn_verify": 0.15,
}


def compute_player_score(
    game_white_fide: int | None,
    game_black_fide: int | None,
    video_white_fide: int | None,
    video_black_fide: int | None,
) -> tuple[float, bool]:
    """Score player match. Returns (score, color_correct).

    Both match correct color: 1.0
    Both match swapped color: 0.8
    Only one matches: 0.3
    Neither matches: 0.0
    """
    if not game_white_fide or not game_black_fide:
        return 0.0, True
    if not video_white_fide or not video_black_fide:
        return 0.0, True

    # Correct color assignment
    if game_white_fide == video_white_fide and game_black_fide == video_black_fide:
        return 1.0, True

    # Swapped colors (video got white/black reversed)
    if game_white_fide == video_black_fide and game_black_fide == video_white_fide:
        return 0.8, False

    # One player matches
    fide_set = {video_white_fide, video_black_fide}
    if game_white_fide in fide_set or game_black_fide in fide_set:
        return 0.3, True

    return 0.0, True


def compute_date_score(
    game_date: date | None,
    video_year: int | None,
    video_published: date | None = None,
) -> float:
    """Score date proximity.

    Same day: 1.0, decays linearly over 7 days to 0.
    Falls back to year comparison if no exact date.
    """
    if not game_date:
        return 0.0

    # If we have the video's publication date, use it
    if video_published:
        delta = abs((game_date - video_published).days)
        if delta <= 1:
            return 1.0
        if delta <= 3:
            return 0.8
        if delta <= 7:
            return 0.5
        if delta <= 30:
            return 0.2
        return 0.0

    # Fall back to year comparison
    if video_year and game_date.year == video_year:
        return 0.5

    return 0.0


def compute_round_score(
    game_round: str | None,
    video_round: str | None,
) -> float:
    """Score round match. 1.0 for exact match, 0.0 otherwise."""
    if not game_round or not video_round:
        return 0.0

    # Normalize: "4.1" matches "4", "R4" matches "4"
    def normalize_round(r: str) -> str:
        r = r.strip().lower()
        r = r.replace("round", "").replace("rd", "").replace("r", "").strip()
        # Take the major round number (e.g., "4.1" -> "4")
        if "." in r:
            r = r.split(".")[0]
        return r

    return 1.0 if normalize_round(game_round) == normalize_round(video_round) else 0.0


def compute_result_score(
    game_result: str | None,
    video_result: str | None,
) -> float:
    """Score result match."""
    if not game_result or not video_result:
        return 0.0

    def normalize_result(r: str) -> str:
        r = r.strip()
        if r in ("½-½", "draw"):
            return "1/2-1/2"
        return r

    return 1.0 if normalize_result(game_result) == normalize_result(video_result) else 0.0


def compute_confidence(
    signals: MatchSignals,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute overall match confidence from individual signals."""
    w = weights or DEFAULT_WEIGHTS

    score = (
        signals.player_match * w["player_match"]
        + signals.event_similarity * w["event_similarity"]
        + signals.date_proximity * w["date_proximity"]
        + signals.round_match * w["round_match"]
        + signals.result_match * w["result_match"]
        + signals.pgn_verify * w["pgn_verify"]
    )

    # Normalize to 0-100 scale
    return min(score * 100, 100.0)
