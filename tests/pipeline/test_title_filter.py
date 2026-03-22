"""Tests for pipeline.screen.title_filter."""

import pytest

from pipeline.screen.title_filter import score_title


class TestPositivePatterns:
    """Titles with OTB chess signals should be candidates."""

    def test_vs_pattern(self):
        ok, conf = score_title("Carlsen vs Nepomniachtchi | World Championship 2023")
        assert ok is True
        assert conf > 0.0

    def test_v_dot_pattern(self):
        ok, _ = score_title("Firouzja v. Caruana Round 5")
        assert ok is True

    def test_versus_pattern(self):
        ok, _ = score_title("Nakamura versus Ding Liren")
        assert ok is True

    def test_tournament_keyword(self):
        ok, _ = score_title("Candidates Tournament 2024 Round 8")
        assert ok is True

    def test_sinquefield_cup(self):
        ok, _ = score_title("Sinquefield Cup 2023 Highlights")
        assert ok is True

    def test_tata_steel(self):
        ok, _ = score_title("Tata Steel Chess 2024")
        assert ok is True

    def test_world_championship(self):
        ok, _ = score_title("World Championship Game 10")
        assert ok is True

    def test_olympiad(self):
        ok, _ = score_title("Chess Olympiad Budapest 2024")
        assert ok is True

    def test_player_vs_player_pattern(self):
        """Two capitalized names separated by vs — strong signal."""
        ok, conf = score_title("Magnus Carlsen vs Ian Nepomniachtchi")
        assert ok is True
        assert conf >= 0.5


class TestNegativePatterns:
    """Titles matching negative keywords should be rejected."""

    def test_puzzle(self):
        ok, _ = score_title("Mate in 3 Puzzle Challenge")
        assert ok is False

    def test_lesson(self):
        ok, _ = score_title("Chess Lesson: The Sicilian Defense")
        assert ok is False

    def test_tutorial(self):
        ok, _ = score_title("Beginner Tutorial: How to Play Chess")
        assert ok is False

    def test_opening_theory(self):
        ok, _ = score_title("Opening Theory: The Ruy Lopez Deep Dive")
        assert ok is False

    def test_top_10(self):
        ok, _ = score_title("Top 10 Best Chess Moves Ever")
        assert ok is False

    def test_guess_the_elo(self):
        ok, _ = score_title("Guess The Elo Episode 47")
        assert ok is False

    def test_how_to(self):
        ok, _ = score_title("How to Win Every Chess Game")
        assert ok is False


class TestClickbaitAndSensationalism:
    """Clickbait, sensationalized, and reaction content should be rejected."""

    def test_all_caps_title(self):
        ok, _ = score_title("MAGNUS CARLSEN DESTROYS EVERYONE AT THE TOURNAMENT")
        assert ok is False

    def test_destroys_keyword(self):
        ok, _ = score_title("Carlsen Destroys Opponent in 20 Moves")
        assert ok is False

    def test_insane_keyword(self):
        ok, _ = score_title("Insane Game Between Two GMs")
        assert ok is False

    def test_excessive_exclamation_marks(self):
        ok, _ = score_title("What a move by Carlsen!!!")
        assert ok is False

    def test_excessive_question_marks(self):
        ok, _ = score_title("Is this the best game ever???")
        assert ok is False

    def test_clickbait_phrase(self):
        ok, _ = score_title("You Won't Believe What Carlsen Did")
        assert ok is False

    def test_wait_for_it(self):
        ok, _ = score_title("Incredible Sacrifice... Wait For It")
        assert ok is False

    def test_first_person_reference(self):
        ok, _ = score_title("I Got Destroyed by a Grandmaster")
        assert ok is False

    def test_join_me(self):
        ok, _ = score_title("Join Me for Some Chess Today")
        assert ok is False

    def test_player_reaction(self):
        ok, _ = score_title("Hikaru Slams Table After Losing")
        assert ok is False

    def test_secret_rule(self):
        ok, _ = score_title("The Secret Rule That Changes Everything")
        assert ok is False

    def test_cursed(self):
        ok, _ = score_title("The Most Cursed Chess Opening")
        assert ok is False

    def test_rizz(self):
        ok, _ = score_title("Chess Rizz Compilation")
        assert ok is False


class TestQuestionTitles:
    """Questions without tournament context should be rejected."""

    def test_vague_question(self):
        ok, _ = score_title("Can You Spot the Winning Move?")
        assert ok is False

    def test_question_with_tournament_context(self):
        """Questions with tournament context should still pass."""
        ok, _ = score_title("Will Carlsen Win the Candidates 2024?")
        assert ok is True

    def test_do_you_see_it(self):
        ok, _ = score_title("Do You See It? The Brilliant Move")
        assert ok is False


class TestFormalTournamentFormat:
    """Formal tournament formats should be accepted."""

    def test_double_pipe_format(self):
        """Player vs Player || Tournament (Year) format."""
        ok, _ = score_title("Carlsen vs Nepomniachtchi || Candidates 2024")
        assert ok is True

    def test_formal_with_year(self):
        ok, conf = score_title(
            "Firouzja vs Caruana || Sinquefield Cup (2023)"
        )
        assert ok is True
        assert conf >= 0.5

    def test_famous_player_no_context(self):
        """Famous players without tournament context — still passes via vs pattern."""
        ok, _ = score_title("Carlsen vs Hikaru")
        assert ok is True

    def test_famous_player_with_clickbait(self):
        """Famous players with clickbait language — rejected."""
        ok, _ = score_title("Carlsen DESTROYS Hikaru in Insane Game")
        assert ok is False


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_title(self):
        ok, conf = score_title("")
        assert ok is False
        assert conf == 0.0

    def test_whitespace_only(self):
        ok, conf = score_title("   ")
        assert ok is False
        assert conf == 0.0

    def test_non_chess_title(self):
        ok, _ = score_title("Subscribe for more content!")
        assert ok is False

    def test_case_insensitivity_vs(self):
        ok1, _ = score_title("Player A VS Player B")
        ok2, _ = score_title("Player A vs Player B")
        ok3, _ = score_title("Player A Vs Player B")
        assert ok1 == ok2 == ok3

    def test_case_insensitivity_tournament(self):
        ok, _ = score_title("CANDIDATES TOURNAMENT 2024")
        assert ok is True

    def test_short_all_caps_ok(self):
        """Short ALL CAPS titles (< 5 words) should not be auto-rejected."""
        ok, _ = score_title("CANDIDATES 2024")
        assert ok is True


class TestConfidenceScoring:
    """More signals should produce higher confidence."""

    def test_vs_only_lower_than_vs_plus_tournament(self):
        _, conf_vs = score_title("Someone vs Someone Else")
        _, conf_both = score_title("Carlsen vs Nepomniachtchi | Candidates 2024")
        assert conf_both > conf_vs

    def test_confidence_capped_at_one(self):
        _, conf = score_title(
            "Magnus Carlsen vs Ian Nepomniachtchi | World Championship Round 5 Game"
        )
        assert conf <= 1.0

    def test_negative_overrides_positive(self):
        """Negative keywords should reject even if positive signals exist."""
        ok, _ = score_title("Tutorial: Carlsen vs Nepomniachtchi Analysis")
        assert ok is False
