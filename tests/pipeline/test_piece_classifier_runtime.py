from __future__ import annotations

import numpy as np
from pipeline.overlay.piece_classifier import (
    classify_square_crops,
    preprocess_square_crops,
)
from pipeline.overlay.square_classifier_model import INPUT_SIZE


def test_preprocess_square_crops_outputs_normalized_nchw() -> None:
    crop = np.zeros((32, 40, 3), dtype=np.uint8)
    crop[:, :, 0] = 10
    crop[:, :, 1] = 20
    crop[:, :, 2] = 30

    batch = preprocess_square_crops([crop, crop])

    assert batch.shape == (2, 3, INPUT_SIZE, INPUT_SIZE)
    assert batch.dtype == np.float32
    assert float(batch.min()) >= 0.0
    assert float(batch.max()) <= 1.0
    # BGR -> RGB swap
    assert np.isclose(batch[0, 0, 0, 0], 30 / 255.0)
    assert np.isclose(batch[0, 2, 0, 0], 10 / 255.0)


def test_classify_square_crops_uses_session_logits(monkeypatch) -> None:
    class FakeSession:
        def run(self, _outputs, _inputs):
            logits = np.array(
                [
                    [0.0, 0.1, 0.9],
                    [0.8, 0.1, 0.0],
                ],
                dtype=np.float32,
            )
            return [logits]

    monkeypatch.setattr(
        "pipeline.overlay.piece_classifier._get_session",
        lambda: (FakeSession(), "input"),
    )

    crop = np.zeros((32, 32, 3), dtype=np.uint8)
    assert classify_square_crops([crop, crop]) == [2, 0]
