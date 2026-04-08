"""Tests for calibration evaluation overlay localization behavior."""

from unittest.mock import patch

import numpy as np
from api.services.evaluate import calibration_eval_service
from pipeline.overlay.scanner import OverlayDetection


class TestDetectOverlayForEval:
    """Calibration eval should mirror the auto-calibration detection stack."""

    @patch("api.services.evaluate.calibration_eval_service._grid_scan_frames")
    @patch("api.services.evaluate.calibration_eval_service.detect_overlay_fast")
    def test_prefers_fast_detector_when_it_finds_overlay(
        self,
        mock_detect_fast,
        mock_grid_scan,
    ) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detection = OverlayDetection(
            found=True,
            bbox=(100, 50, 600, 600),
            frame_resolution=(1920, 1080),
        )
        mock_detect_fast.return_value = detection

        result = calibration_eval_service._detect_overlay_for_eval(frame)

        assert result is detection
        mock_grid_scan.assert_not_called()

    @patch("api.services.evaluate.calibration_eval_service._grid_scan_frames")
    @patch("api.services.evaluate.calibration_eval_service.detect_overlay_fast")
    def test_falls_back_to_grid_scan_when_fast_detector_misses(
        self,
        mock_detect_fast,
        mock_grid_scan,
    ) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_detect_fast.return_value = OverlayDetection(
            found=False,
            frame_resolution=(1920, 1080),
        )
        fallback = OverlayDetection(
            found=True,
            bbox=(120, 80, 620, 620),
            frame_resolution=(1920, 1080),
        )
        mock_grid_scan.return_value = (fallback, frame)

        result = calibration_eval_service._detect_overlay_for_eval(frame)

        assert result is fallback
        mock_grid_scan.assert_called_once_with([(frame, "eval")])

    @patch("api.services.evaluate.calibration_eval_service._grid_scan_frames")
    @patch("api.services.evaluate.calibration_eval_service.detect_overlay_fast")
    def test_returns_not_found_when_both_strategies_miss(
        self,
        mock_detect_fast,
        mock_grid_scan,
    ) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_detect_fast.return_value = OverlayDetection(
            found=False,
            frame_resolution=(1280, 720),
        )
        mock_grid_scan.return_value = (None, None)

        result = calibration_eval_service._detect_overlay_for_eval(frame)

        assert result.found is False
        assert result.bbox is None
        assert result.frame_resolution == (1280, 720)
