from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from pipeline.physical.board_localizer import localize_board


def test_localize_board_prefers_otb_yolo(monkeypatch) -> None:
    monkeypatch.setattr(
        "pipeline.physical.board_localizer.detect_otb_yolo",
        lambda *_args, **_kwargs: SimpleNamespace(
            found=True,
            bbox=(10, 20, 30, 40),
            confidence=0.8,
        ),
    )

    localization = localize_board(np.zeros((80, 80, 3), dtype=np.uint8), device="cpu")

    assert localization is not None
    assert localization.method == "otb_yolo"
    assert localization.corners == ((10.0, 20.0), (40.0, 20.0), (40.0, 60.0), (10.0, 60.0))


def test_localize_board_falls_back_to_alternation_search(monkeypatch) -> None:
    monkeypatch.setattr(
        "pipeline.physical.board_localizer.detect_otb_yolo",
        lambda *_args, **_kwargs: SimpleNamespace(found=False, bbox=None, confidence=0.0),
    )

    image = np.zeros((160, 160, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            value = 220 if (row + col) % 2 == 0 else 40
            y1 = 40 + row * 10
            x1 = 40 + col * 10
            image[y1 : y1 + 10, x1 : x1 + 10] = value

    localization = localize_board(image, device="cpu")

    assert localization is not None
    assert localization.method == "alternation_search"
    xs = [point[0] for point in localization.corners]
    ys = [point[1] for point in localization.corners]
    assert min(xs) <= 45.0
    assert max(xs) >= 110.0
    assert min(ys) <= 45.0
    assert max(ys) >= 110.0
