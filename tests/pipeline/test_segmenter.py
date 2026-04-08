"""Tests for video layout segmentation."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from pipeline.overlay.scanner import OverlayDetection, runtime_overlay_check
from pipeline.overlay.segmenter import (
    _bbox_relative_shift,
    _median_bbox,
    segment_video_layouts,
)


class TestBboxRelativeShift:
    def test_identical_bboxes(self):
        assert _bbox_relative_shift((100, 100, 200, 200), (100, 100, 200, 200)) == 0.0

    def test_shifted_x(self):
        # Shift x by 20 with width 200 -> 20/200 = 0.1
        shift = _bbox_relative_shift((100, 100, 200, 200), (120, 100, 200, 200))
        assert abs(shift - 0.1) < 0.01

    def test_large_shift_flags_layout_change(self):
        # Shift by 50% of width -> 0.5, should be above threshold
        shift = _bbox_relative_shift((100, 100, 200, 200), (200, 100, 200, 200))
        assert shift >= 0.15

    def test_small_shift_within_threshold(self):
        # Shift by 5% -> should be below 0.15 threshold
        shift = _bbox_relative_shift((100, 100, 200, 200), (110, 100, 200, 200))
        assert shift < 0.15

    def test_size_change(self):
        # Width changes from 200 to 260 -> 60/260 ~ 0.23
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


class TestRuntimeOverlayCheck:
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

        det = runtime_overlay_check(frame)
        assert det.found is True
        assert det.bbox is not None
        assert det.score > 0.5

    def test_rejects_random_noise(self):
        """Random noise should not be detected as an overlay."""
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (600, 800, 3), dtype=np.uint8)
        det = runtime_overlay_check(frame)
        assert det.found is False

    def test_downscaling_preserves_detection(self):
        """Overlay detection works on large frames via internal downscaling."""
        size = 960
        frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        cell = size // 8
        for r in range(8):
            for c in range(8):
                color = 220 if (r + c) % 2 == 0 else 80
                y0 = 60 + r * cell
                x0 = 480 + c * cell
                frame[y0 : y0 + cell, x0 : x0 + cell] = color

        det = runtime_overlay_check(frame)
        assert det.found is True
        assert det.bbox is not None
        assert det.bbox[2] > 400


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

    @patch("pipeline.overlay.segmenter.runtime_overlay_check")
    @patch("cv2.VideoCapture")
    def test_all_overlay_produces_single_segment(self, mock_cap_cls, mock_runtime_check):
        """All samples with overlay produce a single merged segment."""
        mock_cap_cls.return_value = self._mock_cap()

        bbox = (500, 200, 600, 600)
        mock_runtime_check.return_value = OverlayDetection(
            found=True,
            bbox=bbox,
            seed_bbox=bbox,
            score=0.85,
            frame_resolution=(1920, 1080),
        )

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 1
        assert segments[0].start_time == 0.0
        assert len(gaps) == 0

    @patch("pipeline.overlay.segmenter.runtime_overlay_check")
    @patch("cv2.VideoCapture")
    def test_all_gap_produces_single_gap(self, mock_cap_cls, mock_runtime_check):
        """All samples without overlay produce a single gap."""
        mock_cap_cls.return_value = self._mock_cap()

        mock_runtime_check.return_value = OverlayDetection(
            found=False,
            frame_resolution=(1920, 1080),
        )

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 0
        assert len(gaps) == 1

    @patch("pipeline.overlay.segmenter.runtime_overlay_check")
    @patch("cv2.VideoCapture")
    def test_overlay_gap_overlay_sequence(self, mock_cap_cls, mock_runtime_check):
        """Overlay -> gap -> overlay produces correct segments and gap."""
        mock_cap_cls.return_value = self._mock_cap(total_frames=1800)

        bbox = (500, 200, 600, 600)

        def check_side_effect(frame):
            call_num = mock_runtime_check.call_count
            t = (call_num - 1) * 5.0
            if 20.0 <= t < 35.0:
                return OverlayDetection(found=False, frame_resolution=(1920, 1080))
            return OverlayDetection(
                found=True,
                bbox=bbox,
                seed_bbox=bbox,
                score=0.85,
                frame_resolution=(1920, 1080),
            )

        mock_runtime_check.side_effect = check_side_effect

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 2
        assert len(gaps) == 1
        assert segments[0].start_time == 0.0
        assert segments[1].end_time <= 60.0

    @patch("pipeline.overlay.segmenter.runtime_overlay_check")
    @patch("cv2.VideoCapture")
    def test_min_overlay_fraction_filter(self, mock_cap_cls, mock_runtime_check):
        """Segments with sub-threshold overlays are treated as gaps."""
        mock_cap_cls.return_value = self._mock_cap(height=1080)

        small_bbox = (500, 340, 400, 400)
        mock_runtime_check.return_value = OverlayDetection(
            found=True,
            bbox=small_bbox,
            seed_bbox=small_bbox,
            score=0.7,
            frame_resolution=(1920, 1080),
        )

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 0
        assert len(gaps) == 1

    @patch("pipeline.overlay.segmenter.runtime_overlay_check")
    @patch("cv2.VideoCapture")
    def test_merges_overlay_with_small_bbox_jitter(self, mock_cap_cls, mock_runtime_check):
        """Consecutive overlay samples merge when bbox jitter is small."""
        mock_cap_cls.return_value = self._mock_cap()

        def jittery_bbox(frame):
            n = mock_runtime_check.call_count
            jitter = 20 if n % 2 == 0 else -20
            bbox = (500 + jitter, 200 + jitter, 600, 600)
            return OverlayDetection(
                found=True,
                bbox=bbox,
                seed_bbox=bbox,
                score=0.8,
                frame_resolution=(1920, 1080),
            )

        mock_runtime_check.side_effect = jittery_bbox

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 1
        assert len(gaps) == 0

    @patch("pipeline.overlay.segmenter.runtime_overlay_check")
    @patch("cv2.VideoCapture")
    def test_splits_overlay_on_large_bbox_shift(self, mock_cap_cls, mock_runtime_check):
        """Overlay segments split when bbox shifts significantly (layout change)."""
        mock_cap_cls.return_value = self._mock_cap(total_frames=1800)

        bbox_left = (100, 200, 600, 600)
        bbox_right = (900, 200, 600, 600)

        def layout_change(frame):
            n = mock_runtime_check.call_count
            t = (n - 1) * 5.0
            bbox = bbox_left if t < 30.0 else bbox_right
            return OverlayDetection(
                found=True,
                bbox=bbox,
                seed_bbox=bbox,
                score=0.8,
                frame_resolution=(1920, 1080),
            )

        mock_runtime_check.side_effect = layout_change

        segments, gaps = segment_video_layouts("/fake/video.mp4")

        assert len(segments) == 2
        assert len(gaps) == 0
