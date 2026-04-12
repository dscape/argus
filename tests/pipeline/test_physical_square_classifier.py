from __future__ import annotations

import numpy as np
from pipeline.physical.square_classifier import _class_ids_to_board_fen


def test_class_ids_to_board_fen_encodes_empty_runs() -> None:
    class_ids = [0] * 64
    class_ids[0] = 10  # black rook on a8
    class_ids[63] = 4  # white rook on h1

    fen = _class_ids_to_board_fen(class_ids)

    assert fen == "r7/8/8/8/8/8/8/7R"


def test_read_fen_from_frame_returns_none_without_weights(monkeypatch) -> None:
    import pipeline.physical.square_classifier as square_classifier

    monkeypatch.setattr(
        square_classifier,
        "_resolve_weights_path",
        lambda: (_ for _ in ()).throw(FileNotFoundError()),
    )

    result = square_classifier.read_fen_from_frame(np.zeros((64, 64, 3), dtype=np.uint8))

    assert result is None
