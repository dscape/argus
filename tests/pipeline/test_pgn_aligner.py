"""Tests for pipeline.clips.pgn_aligner."""

import pytest
from pipeline.clips.pgn_aligner import AlignedMove, AlignmentResult, align_pgn_to_detections


SCHOLARS_MATE = "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0"
ITALIAN_OPENING = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5"


def _make_detections(n: int, fps: float = 30.0, score: float = 0.05) -> list[dict]:
    """Create n synthetic move detections spaced evenly."""
    return [
        {
            "frame_idx": (i + 1) * int(fps * 5),  # every 5 seconds
            "timestamp_seconds": (i + 1) * 5.0,
            "score": score,
        }
        for i in range(n)
    ]


# ── Basic alignment ──────────────────────────────────────────


class TestAlignPgnToDetections:
    def test_perfect_alignment(self):
        """When detected moves == PGN moves, all should align."""
        detections = _make_detections(7)  # Scholar's mate has 7 half-moves
        result = align_pgn_to_detections(SCHOLARS_MATE, detections, 30.0)
        assert result.total_pgn_moves == 7
        assert result.aligned_count == 7
        assert result.error_count == 0
        assert result.quality == pytest.approx(1.0)

    def test_more_detections_than_moves(self):
        """Extra detections are ignored; PGN moves should all align."""
        detections = _make_detections(20)
        result = align_pgn_to_detections(SCHOLARS_MATE, detections, 30.0)
        assert result.aligned_count == 7
        assert result.total_detected_moves == 20

    def test_fewer_detections_than_moves(self):
        """When detections run out, remaining PGN moves are unaligned."""
        detections = _make_detections(3)
        result = align_pgn_to_detections(SCHOLARS_MATE, detections, 30.0)
        assert result.aligned_count == 3
        assert result.total_pgn_moves == 7
        assert result.quality < 1.0

    def test_quality_penalised_for_few_alignments(self):
        detections = _make_detections(2)
        result = align_pgn_to_detections(SCHOLARS_MATE, detections, 30.0)
        assert result.quality == pytest.approx(2 / 7)


# ── Error detection ──────────────────────────────────────────


class TestAlignmentErrors:
    def test_low_score_detections_flagged_as_errors(self):
        """Detections with very low motion score are flagged."""
        detections = _make_detections(7, score=0.001)  # below 0.003 threshold
        result = align_pgn_to_detections(SCHOLARS_MATE, detections, 30.0)
        assert result.error_count == 7
        assert all(m.is_error for m in result.moves)

    def test_high_score_detections_not_errors(self):
        detections = _make_detections(7, score=0.05)
        result = align_pgn_to_detections(SCHOLARS_MATE, detections, 30.0)
        assert result.error_count == 0
        assert not any(m.is_error for m in result.moves)

    def test_quality_penalised_for_errors(self):
        good = _make_detections(7, score=0.05)
        bad = _make_detections(7, score=0.001)
        result_good = align_pgn_to_detections(SCHOLARS_MATE, good, 30.0)
        result_bad = align_pgn_to_detections(SCHOLARS_MATE, bad, 30.0)
        assert result_good.quality > result_bad.quality


# ── Aligned move contents ────────────────────────────────────


class TestAlignedMoveContents:
    def test_move_uci_and_san(self):
        detections = _make_detections(6)  # Italian opening has 6 half-moves
        result = align_pgn_to_detections(ITALIAN_OPENING, detections, 30.0)
        moves = result.moves
        assert moves[0].move_uci == "e2e4"
        assert moves[0].move_san == "e4"
        assert moves[1].move_uci == "e7e5"
        assert moves[1].move_san == "e5"

    def test_fen_before_and_after(self):
        detections = _make_detections(6)
        result = align_pgn_to_detections(ITALIAN_OPENING, detections, 30.0)
        first = result.moves[0]
        # Before first move: starting position
        assert "rnbqkbnr/pppppppp" in first.fen_before
        # After e4: pawn on e4
        assert "4P3" in first.fen_after or "e4" in first.move_uci

    def test_frame_idx_and_timestamp(self):
        detections = _make_detections(6, fps=30.0)
        result = align_pgn_to_detections(ITALIAN_OPENING, detections, 30.0)
        assert result.moves[0].frame_idx == 150  # 5s * 30fps
        assert result.moves[0].timestamp_seconds == 5.0
        assert result.moves[1].frame_idx == 300

    def test_move_indices_sequential(self):
        detections = _make_detections(6)
        result = align_pgn_to_detections(ITALIAN_OPENING, detections, 30.0)
        for i, m in enumerate(result.moves):
            assert m.move_index == i


# ── Edge cases ───────────────────────────────────────────────


class TestAlignmentEdgeCases:
    def test_empty_pgn(self):
        result = align_pgn_to_detections("", _make_detections(5), 30.0)
        assert result.total_pgn_moves == 0
        assert result.aligned_count == 0
        assert result.quality == 0.0

    def test_invalid_pgn(self):
        result = align_pgn_to_detections("not valid pgn", _make_detections(5), 30.0)
        assert result.aligned_count == 0

    def test_no_detections(self):
        result = align_pgn_to_detections(SCHOLARS_MATE, [], 30.0)
        assert result.total_pgn_moves == 7
        assert result.aligned_count == 0
        assert result.quality == 0.0

    def test_both_empty(self):
        result = align_pgn_to_detections("", [], 30.0)
        assert result.quality == 0.0
