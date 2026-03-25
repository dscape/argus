"""Tests for pipeline.screen.screen_pipeline (unit-level, no DB)."""

from pipeline.screen.title_filter import score_title


class TestTitleFilterIntegration:
    """Verify title filter behavior on realistic video titles."""

    def test_batch_of_real_titles(self):
        """Simulate filtering a batch of mixed titles."""
        titles = [
            ("Carlsen vs Nepomniachtchi | WCC 2021 Game 6", True),
            ("How to Checkmate with a Rook", False),
            ("Sinquefield Cup Round 3 Recap", True),
            ("Top 10 Chess Openings for Beginners", False),
            ("Firouzja v. Caruana Candidates 2024", True),
            ("Stream Highlights from Saturday", False),
            ("Tata Steel Chess 2024 Round 7", True),
            ("Guess The Elo #47", False),
        ]

        for title, expected_match in titles:
            ok, _ = score_title(title)
            assert ok == expected_match, (
                f"Title '{title}' expected match={expected_match}, got {ok}"
            )

    def test_all_matches_have_positive_confidence(self):
        """Matched titles should always have confidence > 0."""
        match_titles = [
            "Player A vs Player B",
            "World Championship 2024",
            "Olympiad Round 5",
        ]
        for title in match_titles:
            ok, conf = score_title(title)
            if ok:
                assert conf > 0.0, f"Matched title '{title}' has zero confidence"
