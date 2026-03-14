"""Tests for pipeline.extract.title_parser."""

import pytest
from pipeline.extract.title_parser import (
    TitleExtraction,
    _clean_name,
    _strip_titles,
    parse_chapter_title,
    parse_title,
)


# ── _strip_titles ────────────────────────────────────────────


class TestStripTitles:
    def test_strips_gm(self):
        assert _strip_titles("GM Carlsen") == "Carlsen"

    def test_strips_im(self):
        assert _strip_titles("IM Naroditsky") == "Naroditsky"

    def test_strips_wgm(self):
        assert _strip_titles("WGM Ju Wenjun") == "Ju Wenjun"

    def test_strips_fm_with_dot(self):
        assert _strip_titles("FM. Smith") == "Smith"

    def test_case_insensitive(self):
        assert _strip_titles("gm Carlsen") == "Carlsen"

    def test_no_title(self):
        assert _strip_titles("Carlsen") == "Carlsen"

    def test_multiple_titles(self):
        assert _strip_titles("GM IM Carlsen") == "Carlsen"


# ── _clean_name ──────────────────────────────────────────────


class TestCleanName:
    def test_strips_title_prefix(self):
        assert _clean_name("GM Magnus Carlsen") == "Magnus Carlsen"

    def test_removes_rating_parenthetical(self):
        assert _clean_name("Carlsen (2840)") == "Carlsen"

    def test_removes_trailing_rating(self):
        assert _clean_name("Carlsen 2840") == "Carlsen"

    def test_preserves_normal_name(self):
        assert _clean_name("Magnus Carlsen") == "Magnus Carlsen"

    def test_strips_whitespace(self):
        assert _clean_name("  Carlsen  ") == "Carlsen"


# ── parse_title: "vs" pattern ────────────────────────────────


class TestParseTitleVsPattern:
    def test_simple_vs(self):
        r = parse_title("Carlsen vs Nepomniachtchi")
        assert r.white == "Carlsen"
        assert r.black == "Nepomniachtchi"
        assert r.confidence >= 0.7

    def test_vs_with_dot(self):
        r = parse_title("Carlsen vs. Nepomniachtchi")
        assert r.white == "Carlsen"
        assert r.black == "Nepomniachtchi"

    def test_versus_spelled_out(self):
        r = parse_title("Carlsen versus Nepomniachtchi")
        assert r.white == "Carlsen"
        assert r.black == "Nepomniachtchi"

    def test_vs_with_pipe_event(self):
        r = parse_title("Carlsen vs Nepomniachtchi | World Championship 2021 Round 6")
        assert r.white == "Carlsen"
        assert r.black == "Nepomniachtchi"
        assert r.event == "World Championship"
        assert r.year == 2021
        assert r.round == "6"
        assert r.confidence >= 0.9

    def test_vs_with_dash_event(self):
        r = parse_title("Carlsen vs Nepomniachtchi - Sinquefield Cup 2019")
        assert r.white == "Carlsen"
        assert r.black == "Nepomniachtchi"
        assert r.year == 2019

    def test_gm_title_stripped(self):
        r = parse_title("GM Magnus Carlsen vs GM Ian Nepomniachtchi")
        assert r.white == "Magnus Carlsen"
        assert r.black == "Ian Nepomniachtchi"

    def test_result_extracted(self):
        r = parse_title("Caruana vs Ding Liren | Candidates 2022 Round 14 1-0")
        assert r.result == "1-0"
        assert r.white == "Caruana"
        assert r.black == "Ding Liren"
        assert r.event == "Candidates"
        assert r.year == 2022
        assert r.round == "14"

    def test_draw_result(self):
        r = parse_title("Carlsen vs Firouzja 1/2-1/2")
        assert r.result == "1/2-1/2"

    def test_draw_half_symbol(self):
        r = parse_title("Carlsen vs Firouzja ½-½")
        assert r.result == "1/2-1/2"

    def test_draw_word(self):
        r = parse_title("Carlsen vs Firouzja draw")
        assert r.result == "1/2-1/2"

    def test_black_wins(self):
        r = parse_title("Carlsen vs Firouzja 0-1")
        assert r.result == "0-1"

    def test_result_not_in_event(self):
        """The result '1-0' should not bleed into the event string."""
        r = parse_title("Caruana vs Ding Liren | Candidates 2022 1-0")
        assert r.event is not None
        assert "1" not in r.event and "0" not in r.event

    def test_prefix_event_before_vs(self):
        r = parse_title("Tata Steel 2024 | Carlsen vs Giri")
        assert r.white == "Carlsen"
        assert r.black == "Giri"
        assert "Tata Steel" in r.event
        assert r.year == 2024

    def test_full_names_with_event_round_result(self):
        r = parse_title(
            "Bobby Fischer vs Boris Spassky | 1972 World Chess Championship Round 1"
        )
        assert r.white == "Bobby Fischer"
        assert r.black == "Boris Spassky"
        assert r.year == 1972
        assert r.round == "1"


# ── parse_title: dash-separator fallback ─────────────────────


class TestParseTitleDashFallback:
    def test_dash_separator_low_confidence(self):
        r = parse_title("Kasparov - Karpov")
        assert r.white == "Kasparov"
        assert r.black == "Karpov"
        assert r.confidence <= 0.5

    def test_rejects_chess_keywords(self):
        """Titles containing keywords like 'chess' or 'opening' should not
        be parsed as player names."""
        r = parse_title("Chess Opening - Analysis")
        assert r.white is None
        assert r.black is None


# ── parse_title: year extraction ─────────────────────────────


class TestParseTitleYear:
    def test_extracts_four_digit_year(self):
        r = parse_title("Carlsen vs Giri | Tata Steel 2024")
        assert r.year == 2024

    def test_year_in_1900s(self):
        r = parse_title("Fischer vs Spassky 1972")
        assert r.year == 1972

    def test_no_year(self):
        r = parse_title("Carlsen vs Giri")
        assert r.year is None


# ── parse_title: round extraction ────────────────────────────


class TestParseTitleRound:
    def test_round_n(self):
        r = parse_title("Carlsen vs Giri Round 6")
        assert r.round == "6"

    def test_rd_n(self):
        r = parse_title("Carlsen vs Giri Rd 3")
        assert r.round == "3"

    def test_r_n(self):
        r = parse_title("Carlsen vs Giri R5")
        assert r.round == "5"

    def test_round_with_board(self):
        r = parse_title("Carlsen vs Giri Round 4.2")
        assert r.round == "4.2"

    def test_no_round(self):
        r = parse_title("Carlsen vs Giri")
        assert r.round is None


# ── parse_title: edge cases ──────────────────────────────────


class TestParseTitleEdgeCases:
    def test_empty_string(self):
        r = parse_title("")
        assert r.white is None
        assert r.black is None
        assert r.confidence == 0.0

    def test_no_players(self):
        r = parse_title("Chess Tournament Highlights 2024")
        assert r.confidence == 0.0

    def test_against_keyword(self):
        r = parse_title("Carlsen against Giri")
        assert r.white == "Carlsen"
        assert r.black == "Giri"

    def test_v_abbreviation(self):
        r = parse_title("Carlsen v Giri")
        assert r.white == "Carlsen"
        assert r.black == "Giri"

    def test_confidence_scales_with_info(self):
        """More extracted fields → higher confidence."""
        r_minimal = parse_title("Carlsen vs Giri")
        r_with_year = parse_title("Carlsen vs Giri 2024")
        r_full = parse_title(
            "Carlsen vs Giri | Tata Steel 2024 Round 5 1-0"
        )
        assert r_minimal.confidence < r_with_year.confidence
        assert r_with_year.confidence < r_full.confidence


# ── parse_chapter_title ──────────────────────────────────────


class TestParseChapterTitle:
    def test_simple_chapter(self):
        r = parse_chapter_title("Carlsen vs Nepomniachtchi")
        assert r.white == "Carlsen"
        assert r.black == "Nepomniachtchi"

    def test_chapter_delegates_to_parse_title(self):
        """Chapter parser uses the same logic as title parser."""
        r = parse_chapter_title("GM Caruana vs GM Ding Liren")
        assert r.white == "Caruana"
        assert r.black == "Ding Liren"
