"""Tests for default YOLO detector routing."""

from __future__ import annotations

import numpy as np
import pipeline.overlay.scanner as scanner
from pipeline.overlay.scanner import OverlayDetection
from pipeline.overlay.yolo_detector import DEFAULT_WEIGHTS_PATH


def test_default_overlay_weights_are_committed() -> None:
    assert DEFAULT_WEIGHTS_PATH.exists()


def test_runtime_overlay_check_uses_default_yolo(monkeypatch) -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    expected = OverlayDetection(found=True, bbox=(1, 2, 3, 4), score=0.9)
    monkeypatch.setattr(scanner, "_run_yolo_overlay_detector", lambda _: expected)

    assert scanner.runtime_overlay_check(frame) == expected


def test_detect_overlay_runtime_pads_default_yolo_bbox(monkeypatch) -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    expected = OverlayDetection(
        found=True,
        bbox=(10, 10, 20, 20),
        seed_bbox=(10, 10, 20, 20),
        score=0.9,
        frame_resolution=(64, 64),
    )
    monkeypatch.setattr(scanner, "_run_yolo_overlay_detector", lambda _: expected)

    assert scanner.detect_overlay_runtime(frame) == OverlayDetection(
        found=True,
        bbox=(2, 2, 36, 36),
        seed_bbox=(10, 10, 20, 20),
        score=0.9,
        frame_resolution=(64, 64),
    )
