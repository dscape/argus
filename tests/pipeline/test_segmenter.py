"""Tests for video layout segmentation."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from pipeline.overlay.scanner import OverlayDetection, fast_overlay_check
from pipeline.overlay.scene_detector import SceneBoundary
from pipeline.overlay.segmenter import (
    _bbox_relative_shift,
    _median_bbox,
    segment_video_layouts,
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


class TestFastOverlayCheck:
    def test_detects_synthetic_rendered_board(self):
        """A synthetic 8x8 alternating-color grid should be detected."""
        size = 400
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        cell = size // 8
        for r in range(8):
            for c in range(8):
                color = 220 if (r + c) % 2 == 0 else 80
                y0 = 100 + r * cell
                x0 = 200 + c * cell
                frame[y0 : y0 + cell, x0 : x0 + cell] = color

        det = fast_overlay_check(frame)
        assert det.found is True
        assert det.bbox is not None
        assert det.score > 0.5

    def test_rejects_random_noise(self):
        """Random noise should not be detected as an overlay."""
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (600, 800, 3), dtype=np.uint8)
        det = fast_overlay_check(frame)
        assert det.found is False


class TestSegmentVideoLayouts:
    """Integration tests for the full pipeline with mocked dependencies."""

    def _mock_cap(self, width=1920, height=1080, fps=30.0, total_frames=108000):
        """Create a mock cv2.VideoCapture that returns video metadata."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((height, width, 3), dtype=np.uint8))

        def get_prop(prop):
            return {
                cv2.CAP_PROP_POS_MSEC: 0,
                cv2.CAP_PROP_FPS: fps,
                cv2.CAP_PROP_FRAME_WIDTH: float(width),
                cv2.CAP_PROP_FRAME_HEIGHT: float(height),
                cv2.CAP_PROP_FRAME_COUNT: float(total_frames),
            }.get(prop, 0)

        cap.get.side_effect = get_prop
        return cap

    @patch("pipeline.overlay.segmenter.detect_scenes")
    @patch("pipeline.overlay.segmenter.fast_overlay_check")
    @patch("cv2.VideoCapture")
    def test_merges_consecutive_overlay_scenes(
        self, mock_cap_cls, mock_fast_check, mock_detect_scenes
    ):
        """Adjacent scenes with similar overlay bboxes merge into one segment."""
        mock_cap_cls.return_value = self._mock_cap()

        mock_detect_scenes.return_value = [
            SceneBoundary(0, 900, 0.0, 30.0),
            SceneBoundary(900, 1800, 30.0, 60.0),
            SceneBoundary(1800, 2700, 60.0, 90.0),
        ]

        bbox = (500, 200, 600, 600)
        mock_fast_check.return_value = OverlayDetection(
            found=True,
            bbox=bbox,
            seed_bbox=bbox,
            score=0.85,
            frame_resolution=(1920, 1080),
        )

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 1
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 90.0
        assert segments[0].sample_count == 3
        assert len(gaps) == 0

    @patch("pipeline.overlay.segmenter.detect_scenes")
    @patch("pipeline.overlay.segmenter.fast_overlay_check")
    @patch("cv2.VideoCapture")
    def test_merges_consecutive_gaps(self, mock_cap_cls, mock_fast_check, mock_detect_scenes):
        """Adjacent scenes without overlay merge into one gap."""
        mock_cap_cls.return_value = self._mock_cap()

        mock_detect_scenes.return_value = [
            SceneBoundary(0, 900, 0.0, 30.0),
            SceneBoundary(900, 1800, 30.0, 60.0),
        ]

        mock_fast_check.return_value = OverlayDetection(
            found=False,
            frame_resolution=(1920, 1080),
        )

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 0
        assert len(gaps) == 1
        assert gaps[0] == (0.0, 60.0)

    @patch("pipeline.overlay.segmenter.detect_scenes")
    @patch("pipeline.overlay.segmenter.fast_overlay_check")
    @patch("cv2.VideoCapture")
    def test_merges_overlay_scenes_regardless_of_bbox(
        self, mock_cap_cls, mock_fast_check, mock_detect_scenes
    ):
        """Consecutive overlay scenes merge even with different bboxes.

        fast_overlay_check returns approximate sub-regions so bbox positions
        vary between scenes. Calibration refines the actual position later.
        """
        mock_cap_cls.return_value = self._mock_cap()

        mock_detect_scenes.return_value = [
            SceneBoundary(0, 2700, 0.0, 90.0),
            SceneBoundary(2700, 5400, 90.0, 180.0),
        ]

        bbox_left = (100, 200, 600, 600)
        bbox_right = (800, 200, 600, 600)

        mock_fast_check.side_effect = [
            OverlayDetection(
                found=True,
                bbox=bbox_left,
                seed_bbox=bbox_left,
                score=0.8,
                frame_resolution=(1920, 1080),
            ),
            OverlayDetection(
                found=True,
                bbox=bbox_right,
                seed_bbox=bbox_right,
                score=0.8,
                frame_resolution=(1920, 1080),
            ),
        ]

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 1
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 180.0
        assert segments[0].sample_count == 2

    @patch("pipeline.overlay.segmenter.detect_scenes")
    @patch("pipeline.overlay.segmenter.fast_overlay_check")
    @patch("cv2.VideoCapture")
    def test_min_overlay_fraction_filter(self, mock_cap_cls, mock_fast_check, mock_detect_scenes):
        """Segments with small overlays are treated as gaps."""
        mock_cap_cls.return_value = self._mock_cap(height=1080)

        mock_detect_scenes.return_value = [
            SceneBoundary(0, 2700, 0.0, 90.0),
        ]

        # Overlay only 200px on a 1080p frame → 200/1080 ≈ 0.19, well below 0.55
        small_bbox = (500, 400, 200, 200)
        mock_fast_check.return_value = OverlayDetection(
            found=True,
            bbox=small_bbox,
            seed_bbox=small_bbox,
            score=0.7,
            frame_resolution=(1920, 1080),
        )

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 0
        assert len(gaps) == 1

    @patch("pipeline.overlay.segmenter.detect_scenes")
    @patch("pipeline.overlay.segmenter.fast_overlay_check")
    @patch("cv2.VideoCapture")
    def test_overlay_then_gap_then_overlay(self, mock_cap_cls, mock_fast_check, mock_detect_scenes):
        """Mixed sequence: overlay → gap → overlay produces correct output."""
        mock_cap_cls.return_value = self._mock_cap()

        mock_detect_scenes.return_value = [
            SceneBoundary(0, 2700, 0.0, 90.0),
            SceneBoundary(2700, 3600, 90.0, 120.0),
            SceneBoundary(3600, 5400, 120.0, 180.0),
        ]

        bbox = (500, 200, 600, 600)
        mock_fast_check.side_effect = [
            OverlayDetection(
                found=True, bbox=bbox, seed_bbox=bbox, score=0.85, frame_resolution=(1920, 1080)
            ),
            OverlayDetection(found=False, frame_resolution=(1920, 1080)),
            OverlayDetection(
                found=True, bbox=bbox, seed_bbox=bbox, score=0.85, frame_resolution=(1920, 1080)
            ),
        ]

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 2
        assert len(gaps) == 1
        assert gaps[0] == (90.0, 120.0)
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 90.0
        assert segments[1].start_time == 120.0
        assert segments[1].end_time == 180.0
