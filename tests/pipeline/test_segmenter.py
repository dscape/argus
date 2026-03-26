"""Tests for video layout segmentation."""

from pipeline.overlay.segmenter import (
    _bbox_relative_shift,
    _median_bbox,
)


class TestBboxRelativeShift:
    def test_identical_bboxes(self):
        assert _bbox_relative_shift((100, 100, 200, 200), (100, 100, 200, 200)) == 0.0

    def test_shifted_x(self):
        # Shift x by 20 with width 200 → 20/200 = 0.1
        shift = _bbox_relative_shift((100, 100, 200, 200), (120, 100, 200, 200))
        assert abs(shift - 0.1) < 0.01

    def test_large_shift_flags_layout_change(self):
        # Shift by 50% of width → 0.5, should be above threshold
        shift = _bbox_relative_shift((100, 100, 200, 200), (200, 100, 200, 200))
        assert shift >= 0.15

    def test_small_shift_within_threshold(self):
        # Shift by 5% → should be below 0.15 threshold
        shift = _bbox_relative_shift((100, 100, 200, 200), (110, 100, 200, 200))
        assert shift < 0.15

    def test_size_change(self):
        # Width changes from 200 to 260 → 60/260 ≈ 0.23
        shift = _bbox_relative_shift((100, 100, 200, 200), (100, 100, 260, 200))
        assert shift > 0.15


class TestMedianBbox:
    def test_single_bbox(self):
        assert _median_bbox([(10, 20, 30, 40)]) == (10, 20, 30, 40)

    def test_multiple_bboxes(self):
        bboxes = [(10, 20, 100, 100), (12, 18, 102, 98), (14, 22, 98, 104)]
        result = _median_bbox(bboxes)
        assert result == (12, 20, 100, 100)

    def test_two_bboxes_averages(self):
        # Median of 2 values is the mean
        result = _median_bbox([(10, 20, 100, 100), (20, 30, 110, 110)])
        assert result == (15, 25, 105, 105)
