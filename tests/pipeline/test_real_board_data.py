from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pipeline.overlay.real_board_data import (
    parse_real_board_fen,
    sample_real_board_squares,
)


def _write_checkerboard(path: str, size: int = 80) -> None:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    square = size // 8
    for row in range(8):
        for col in range(8):
            color = 220 if (row + col) % 2 == 0 else 60
            y1 = row * square
            x1 = col * square
            image[y1 : y1 + square, x1 : x1 + square] = color
    assert cv2.imwrite(path, image)


def test_parse_real_board_fen_frame_filename() -> None:
    filename = "f_lQXos0du0bg_25pct_6k1-1p2pp2-2q4p-2p5-2P5-7P-1PQ1PP2-4R1K1.jpg"
    assert parse_real_board_fen(filename) == "6k1/1p2pp2/2q4p/2p5/2P5/7P/1PQ1PP2/4R1K1"


def test_parse_real_board_fen_runtime_filename() -> None:
    filename = "r21_rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR.jpg"
    assert parse_real_board_fen(filename) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_parse_real_board_fen_applies_known_label_fixup() -> None:
    filename = "f_lQXos0du0bg_25pct_2p1bppp-p7-3pR3-p2P4-1P6-P1P2PPP-RNBQ2K1-8.jpg"
    assert parse_real_board_fen(filename) == "6k1/1p2pp2/2q4p/2p5/2P5/7P/1PQ1PP2/4R1K1"


def test_parse_real_board_fen_for_saved_05zgojs1lsc_sample() -> None:
    path = Path(
        "data/overlay/val_real/"
        "f_05zgojs1Lsc_50pct_1r1q1rk1-4bpp1-p2p1n2-1p2p2p-4P3-P1N4P-1PPQ1PP1-1K1R1B1R.jpg"
    )
    assert path.exists()
    assert parse_real_board_fen(path) == "1r1q1rk1/4bpp1/p2p1n2/1p2p2p/4P3/P1N4P/1PPQ1PP1/1K1R1B1R"


def test_sample_real_board_squares_respects_class_targets(tmp_path) -> None:
    board_path = tmp_path / "r1_8-8-8-8-8-8-8-8.jpg"
    _write_checkerboard(str(board_path))

    images, labels = sample_real_board_squares(tmp_path, max_per_class={0: 10}, size=32, seed=7)

    assert images.shape == (10, 32, 32, 3)
    assert labels.shape == (10,)
    assert labels.tolist() == [0] * 10
