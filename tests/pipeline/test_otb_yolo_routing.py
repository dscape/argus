"""Tests for default OTB YOLO detector routing."""

from __future__ import annotations

import numpy as np
import pipeline.screen.dual_region_detector as dual_region_detector
from pipeline.screen.dual_region_detector import OTBDetection
from pipeline.screen.otb_yolo_detector import DEFAULT_WEIGHTS_PATH


def test_default_otb_weights_are_committed() -> None:
    assert DEFAULT_WEIGHTS_PATH.exists()


def test_detect_otb_region_masks_overlay_before_yolo(monkeypatch) -> None:
    frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    frame[10:30, 20:40] = 123

    def fake_detector(masked: np.ndarray) -> OTBDetection:
        assert np.all(masked[10:30, 20:40] == 0)
        return OTBDetection(
            found=True,
            confidence=0.9,
            bbox=(0, 0, 10, 10),
            frame_resolution=(64, 64),
        )

    monkeypatch.setattr(dual_region_detector, "_run_otb_yolo_detector", fake_detector)

    result = dual_region_detector.detect_otb_region(frame, (20, 10, 20, 20))

    assert result == OTBDetection(
        found=True,
        confidence=0.9,
        bbox=(0, 0, 10, 10),
        frame_resolution=(64, 64),
    )


def test_detect_otb_region_rejects_bbox_overlapping_overlay(monkeypatch) -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay_bbox = (20, 10, 20, 20)
    monkeypatch.setattr(
        dual_region_detector,
        "_run_otb_yolo_detector",
        lambda _: OTBDetection(
            found=True,
            confidence=0.9,
            bbox=(22, 12, 18, 18),
            frame_resolution=(64, 64),
        ),
    )

    result = dual_region_detector.detect_otb_region(frame, overlay_bbox)

    assert result == OTBDetection(found=False, frame_resolution=(64, 64))
