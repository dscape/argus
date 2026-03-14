"""Tests for pipeline.extract.description_parser."""

import pytest
from pipeline.extract.description_parser import (
    Chapter,
    extract_event_from_description,
    extract_pgn,
    parse_chapters,
)


# ── parse_chapters ───────────────────────────────────────────


class TestParseChapters:
    def test_basic_chapters(self):
        desc = "0:00 Introduction\n2:30 Game 1\n5:00 Game 2"
        chapters = parse_chapters(desc)
        assert len(chapters) == 3
        assert chapters[0] == Chapter(0, "Introduction")
        assert chapters[1] == Chapter(150, "Game 1")
        assert chapters[2] == Chapter(300, "Game 2")

    def test_hours_minutes_seconds(self):
        desc = "1:23:10 Carlsen vs Giri"
        chapters = parse_chapters(desc)
        assert len(chapters) == 1
        assert chapters[0].timestamp_seconds == 1 * 3600 + 23 * 60 + 10

    def test_mixed_formats(self):
        desc = "0:00 Intro\n0:45 Game 1\n1:23:10 Game 2\n2:01:00 Wrap up"
        chapters = parse_chapters(desc)
        assert len(chapters) == 4
        assert chapters[0].timestamp_seconds == 0
        assert chapters[1].timestamp_seconds == 45
        assert chapters[2].timestamp_seconds == 4990
        assert chapters[3].timestamp_seconds == 7260

    def test_chapters_with_surrounding_text(self):
        desc = (
            "Great tournament coverage!\n\n"
            "Timestamps:\n"
            "0:00 Introduction\n"
            "5:30 Carlsen vs Nepo\n\n"
            "Thanks for watching!"
        )
        chapters = parse_chapters(desc)
        assert len(chapters) == 2
        assert chapters[0].title == "Introduction"
        assert chapters[1].title == "Carlsen vs Nepo"
        assert chapters[1].timestamp_seconds == 330

    def test_no_chapters(self):
        desc = "This is a regular description with no timestamps."
        assert parse_chapters(desc) == []

    def test_empty_description(self):
        assert parse_chapters("") == []

    def test_preserves_chapter_title(self):
        desc = "0:00 GM Carlsen vs GM Nepomniachtchi | World Championship"
        chapters = parse_chapters(desc)
        assert chapters[0].title == "GM Carlsen vs GM Nepomniachtchi | World Championship"

    def test_two_digit_minutes(self):
        desc = "45:30 Late game analysis"
        chapters = parse_chapters(desc)
        assert len(chapters) == 1
        assert chapters[0].timestamp_seconds == 45 * 60 + 30

    def test_ignores_non_timestamp_numbers(self):
        """Lines that start with text, not timestamps, are ignored."""
        desc = "Check out 1:00 game\n2:30 Actual chapter"
        chapters = parse_chapters(desc)
        assert len(chapters) == 1
        assert chapters[0].title == "Actual chapter"


# ── extract_pgn ──────────────────────────────────────────────


class TestExtractPgn:
    def test_extracts_valid_pgn(self):
        desc = (
            "Analysis of this amazing game:\n\n"
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 "
            "6. Re1 b5 7. Bb3 d6 8. c3 O-O 1-0\n\n"
            "Subscribe for more!"
        )
        pgn = extract_pgn(desc)
        assert pgn is not None
        assert "1. e4" in pgn

    def test_returns_none_for_no_pgn(self):
        desc = "Just a regular description without any chess moves."
        assert extract_pgn(desc) is None

    def test_returns_none_for_too_few_moves(self):
        desc = "1. e4 e5 2. Nf3"
        assert extract_pgn(desc) is None

    def test_ignores_urls_in_pgn_block(self):
        desc = (
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 "
            "https://example.com/game "
            "6. Re1 b5 7. Bb3 d6 1-0"
        )
        pgn = extract_pgn(desc)
        assert pgn is not None
        assert "https" not in pgn

    def test_agadmator_style_description(self):
        """agadmator often pastes full PGN in descriptions."""
        desc = (
            "Carlsen vs Nepomniachtchi\n"
            "World Championship 2021\n\n"
            "1. d4 Nf6 2. Nf3 d5 3. g3 e6 4. Bg2 Be7 5. O-O O-O "
            "6. b3 c5 7. dxc5 Bxc5 8. c4 dxc4 9. bxc4 Qc7 10. Bb2 Nc6 1/2-1/2\n\n"
            "Follow me on Twitter!"
        )
        pgn = extract_pgn(desc)
        assert pgn is not None
        assert "1. d4" in pgn


# ── extract_event_from_description ───────────────────────────


class TestExtractEventFromDescription:
    def test_event_colon(self):
        desc = "Event: Tata Steel Masters 2024\nRound 5"
        assert extract_event_from_description(desc) == "Tata Steel Masters 2024"

    def test_tournament_dash(self):
        desc = "Tournament - Sinquefield Cup 2023\nGame analysis"
        assert extract_event_from_description(desc) == "Sinquefield Cup 2023"

    def test_case_insensitive(self):
        desc = "event: Norway Chess 2024"
        assert extract_event_from_description(desc) == "Norway Chess 2024"

    def test_no_event(self):
        desc = "Just a regular video about chess"
        assert extract_event_from_description(desc) is None

    def test_only_checks_first_five_lines(self):
        desc = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nEvent: Hidden Event"
        assert extract_event_from_description(desc) is None
