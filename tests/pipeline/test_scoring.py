"""Tests for pipeline.match.scoring."""

import pytest
from datetime import date

from pipeline.match.scoring import (
    DEFAULT_WEIGHTS,
    MatchSignals,
    compute_confidence,
    compute_date_score,
    compute_player_score,
    compute_result_score,
    compute_round_score,
)


# ── compute_player_score ─────────────────────────────────────


class TestComputePlayerScore:
    def test_exact_match_correct_color(self):
        score, color = compute_player_score(100, 200, 100, 200)
        assert score == 1.0
        assert color is True

    def test_exact_match_swapped_color(self):
        score, color = compute_player_score(100, 200, 200, 100)
        assert score == 0.8
        assert color is False

    def test_one_player_matches(self):
        score, color = compute_player_score(100, 200, 100, 999)
        assert score == 0.3
        assert color is True

    def test_no_match(self):
        score, color = compute_player_score(100, 200, 300, 400)
        assert score == 0.0

    def test_game_missing_fide_ids(self):
        score, _ = compute_player_score(None, 200, 100, 200)
        assert score == 0.0

    def test_video_missing_fide_ids(self):
        score, _ = compute_player_score(100, 200, None, None)
        assert score == 0.0

    def test_both_none(self):
        score, _ = compute_player_score(None, None, None, None)
        assert score == 0.0

    def test_same_player_both_sides(self):
        """Edge case: same FIDE ID on both sides."""
        score, color = compute_player_score(100, 100, 100, 100)
        assert score == 1.0
        assert color is True


# ── compute_date_score ───────────────────────────────────────


class TestComputeDateScore:
    def test_same_day(self):
        d = date(2024, 1, 15)
        assert compute_date_score(d, None, d) == 1.0

    def test_one_day_apart(self):
        assert compute_date_score(date(2024, 1, 15), None, date(2024, 1, 16)) == 1.0

    def test_three_days(self):
        assert compute_date_score(date(2024, 1, 15), None, date(2024, 1, 18)) == 0.8

    def test_seven_days(self):
        assert compute_date_score(date(2024, 1, 15), None, date(2024, 1, 22)) == 0.5

    def test_thirty_days(self):
        assert compute_date_score(date(2024, 1, 15), None, date(2024, 2, 14)) == 0.2

    def test_beyond_thirty_days(self):
        assert compute_date_score(date(2024, 1, 15), None, date(2024, 6, 1)) == 0.0

    def test_year_only_match(self):
        assert compute_date_score(date(2024, 1, 15), 2024) == 0.5

    def test_year_only_mismatch(self):
        assert compute_date_score(date(2024, 1, 15), 2023) == 0.0

    def test_no_game_date(self):
        assert compute_date_score(None, 2024) == 0.0

    def test_no_video_info(self):
        assert compute_date_score(date(2024, 1, 15), None) == 0.0

    def test_published_date_takes_precedence(self):
        """When both year and published date are given, published date wins."""
        score = compute_date_score(date(2024, 1, 15), 2020, date(2024, 1, 15))
        assert score == 1.0  # Published date matches, year is ignored


# ── compute_round_score ──────────────────────────────────────


class TestComputeRoundScore:
    def test_exact_match(self):
        assert compute_round_score("5", "5") == 1.0

    def test_round_prefix(self):
        assert compute_round_score("5", "Round 5") == 1.0

    def test_rd_prefix(self):
        assert compute_round_score("3", "Rd 3") == 1.0

    def test_r_prefix(self):
        assert compute_round_score("7", "R7") == 1.0

    def test_sub_round_matches_major(self):
        """'4.1' should match '4' (major round number)."""
        assert compute_round_score("4.1", "4") == 1.0

    def test_mismatch(self):
        assert compute_round_score("5", "6") == 0.0

    def test_none_game_round(self):
        assert compute_round_score(None, "5") == 0.0

    def test_none_video_round(self):
        assert compute_round_score("5", None) == 0.0

    def test_both_none(self):
        assert compute_round_score(None, None) == 0.0


# ── compute_result_score ─────────────────────────────────────


class TestComputeResultScore:
    def test_white_wins(self):
        assert compute_result_score("1-0", "1-0") == 1.0

    def test_black_wins(self):
        assert compute_result_score("0-1", "0-1") == 1.0

    def test_draw(self):
        assert compute_result_score("1/2-1/2", "1/2-1/2") == 1.0

    def test_draw_half_symbol(self):
        assert compute_result_score("1/2-1/2", "½-½") == 1.0

    def test_draw_word(self):
        assert compute_result_score("1/2-1/2", "draw") == 1.0

    def test_mismatch(self):
        assert compute_result_score("1-0", "0-1") == 0.0

    def test_none_values(self):
        assert compute_result_score(None, "1-0") == 0.0
        assert compute_result_score("1-0", None) == 0.0


# ── compute_confidence ───────────────────────────────────────


class TestComputeConfidence:
    def test_perfect_match(self):
        signals = MatchSignals(
            player_match=1.0,
            event_similarity=1.0,
            date_proximity=1.0,
            round_match=1.0,
            result_match=1.0,
            pgn_verify=1.0,
        )
        conf = compute_confidence(signals)
        assert conf == 100.0

    def test_zero_signals(self):
        signals = MatchSignals()
        assert compute_confidence(signals) == 0.0

    def test_player_only(self):
        signals = MatchSignals(player_match=1.0)
        conf = compute_confidence(signals)
        assert conf == pytest.approx(35.0)

    def test_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_custom_weights(self):
        signals = MatchSignals(player_match=1.0)
        custom = {k: 0.0 for k in DEFAULT_WEIGHTS}
        custom["player_match"] = 1.0
        assert compute_confidence(signals, custom) == 100.0

    def test_capped_at_100(self):
        """Confidence should never exceed 100."""
        signals = MatchSignals(
            player_match=1.0,
            event_similarity=1.0,
            date_proximity=1.0,
            round_match=1.0,
            result_match=1.0,
            pgn_verify=1.0,
        )
        huge_weights = {k: 2.0 for k in DEFAULT_WEIGHTS}
        assert compute_confidence(signals, huge_weights) == 100.0

    def test_signals_to_dict(self):
        signals = MatchSignals(player_match=0.8, round_match=1.0)
        d = signals.to_dict()
        assert d["player_match"] == 0.8
        assert d["round_match"] == 1.0
        assert d["color_correct"] is True
        assert "player_match" in d
