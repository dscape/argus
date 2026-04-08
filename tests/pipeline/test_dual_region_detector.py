"""Tests for pipeline.screen.dual_region_detector."""

from __future__ import annotations

import numpy as np
import pipeline.screen.dual_region_detector as dual_region_detector
from pipeline.screen.dual_region_detector import OTBDetection, ScreeningResult, detect_otb_region


def test_detect_otb_region_returns_detector_bbox(monkeypatch) -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    expected = OTBDetection(
        found=True,
        confidence=0.82,
        bbox=(120, 340, 640, 480),
        frame_resolution=(1920, 1080),
    )
    monkeypatch.setattr(dual_region_detector, "_run_otb_yolo_detector", lambda _: expected)

    assert detect_otb_region(frame, (1400, 100, 400, 400)) == expected


def test_full_screen_overlay_rejected() -> None:
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    result = detect_otb_region(frame, (10, 10, 480, 480))

    assert result == OTBDetection(found=False, frame_resolution=(500, 500))


class TestScreeningResult:
    """Test the ScreeningResult dataclass logic."""

    def test_approved_when_both_present(self):
        r = ScreeningResult(has_overlay=True, has_otb=True, approved=True)
        assert r.approved is True

    def test_rejected_no_overlay(self):
        r = ScreeningResult(has_overlay=False, has_otb=True, approved=False)
        assert r.approved is False

    def test_rejected_no_otb(self):
        r = ScreeningResult(has_overlay=True, has_otb=False, approved=False)
        assert r.approved is False

    def test_rejected_neither(self):
        r = ScreeningResult(has_overlay=False, has_otb=False, approved=False)
        assert r.approved is False
