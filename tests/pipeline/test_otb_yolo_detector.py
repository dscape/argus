"""Tests for adaptive OTB YOLO retry behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pipeline.screen.otb_yolo_detector as otb_yolo_detector
import pytest


class _FakeBoxes:
    def __init__(self, confs: list[float], xyxys: list[list[float]]) -> None:
        self.conf = np.array(confs, dtype=np.float32)
        self.xyxy = np.array(xyxys, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.conf))


class _FakeModel:
    def __init__(self, responses: dict[int, tuple[list[float], list[list[float]]] | None]) -> None:
        self._responses = responses
        self.calls: list[int] = []

    def predict(self, *, imgsz: int, **_: object) -> list[SimpleNamespace]:
        self.calls.append(imgsz)
        response = self._responses.get(imgsz)
        if response is None:
            return [SimpleNamespace(boxes=_FakeBoxes([], []))]

        confs, xyxys = response
        return [SimpleNamespace(boxes=_FakeBoxes(confs, xyxys))]


def test_detect_otb_yolo_retries_at_higher_resolution_after_miss(monkeypatch) -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    fake_model = _FakeModel({
        640: None,
        1280: ([0.42], [[1321, 878, 1715, 1057]]),
    })

    monkeypatch.setattr(otb_yolo_detector, "_load_model", lambda _: fake_model)
    monkeypatch.setattr(otb_yolo_detector, "_resolve_weights_path", lambda _: Path("fake.pt"))

    result = otb_yolo_detector.detect_otb_yolo(
        frame,
        weights_path="fake.pt",
        imgsz=640,
        device="cpu",
    )

    assert fake_model.calls == [640, 1280]
    assert result.found is True
    assert result.bbox == (1321, 878, 394, 179)
    assert result.frame_resolution == (1920, 1080)
    assert result.confidence == pytest.approx(0.42)


def test_detect_otb_yolo_does_not_retry_after_initial_hit(monkeypatch) -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    fake_model = _FakeModel({640: ([0.91], [[10, 20, 110, 220]])})

    monkeypatch.setattr(otb_yolo_detector, "_load_model", lambda _: fake_model)
    monkeypatch.setattr(otb_yolo_detector, "_resolve_weights_path", lambda _: Path("fake.pt"))

    result = otb_yolo_detector.detect_otb_yolo(
        frame,
        weights_path="fake.pt",
        imgsz=640,
        device="cpu",
    )

    assert fake_model.calls == [640]
    assert result.found is True
    assert result.bbox == (10, 20, 100, 200)
    assert result.frame_resolution == (1280, 720)
    assert result.confidence == pytest.approx(0.91)
