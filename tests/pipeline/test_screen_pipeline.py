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

        for title, expected_candidate in titles:
            ok, _ = score_title(title)
            assert ok == expected_candidate, (
                f"Title '{title}' expected candidate={expected_candidate}, got {ok}"
            )

    def test_all_candidates_have_positive_confidence(self):
        """Candidates should always have confidence > 0."""
        candidate_titles = [
            "Player A vs Player B",
            "World Championship 2024",
            "Olympiad Round 5",
        ]
        for title in candidate_titles:
            ok, conf = score_title(title)
            if ok:
                assert conf > 0.0, f"Candidate '{title}' has zero confidence"
