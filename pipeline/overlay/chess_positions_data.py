"""Data loader for the chess-positions dataset (koryakinp).

Images are 400×400 JPEG with FEN encoded in filenames (hyphens replace slashes).
Each board has a uniform 50×50 grid (no detection needed).

Used to augment synthetic training data with diverse real-world board styles.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import chess
import cv2
import numpy as np

from pipeline.overlay.piece_classifier_data import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHESS_POSITIONS_DIR = _PROJECT_ROOT / "data" / "chess_positions"

BOARD_SIZE = 400
SQ_SIZE = 50

# Piece symbol → class index (matches CLASS_NAMES / CLASS_TO_PIECE)
_PIECE_TO_CLASS = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}


def parse_fen_from_filename(filename: str) -> str:
    """Convert filename to FEN piece-placement string.

    ``'1B1B2K1-1B6-5N2-6k1-8-8-8-4nq2.jpeg'``
    → ``'1B1B2K1/1B6/5N2/6k1/8/8/8/4nq2'``
    """
    stem = Path(filename).stem
    return stem.replace("-", "/")


def fen_to_square_labels(fen: str) -> list[list[int]]:
    """Convert piece-placement FEN to 8×8 grid of class indices (0–12).

    Row 0 = rank 8 (top of board image).  Matches visual layout.
    """
    ranks = fen.split("/")
    assert len(ranks) == 8, f"Expected 8 ranks, got {len(ranks)}: {fen}"

    grid: list[list[int]] = []
    for rank_str in ranks:
        row: list[int] = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend([0] * int(ch))
            else:
                cls = _PIECE_TO_CLASS.get(ch)
                assert cls is not None, f"Unknown piece char: {ch}"
                row.append(cls)
        assert len(row) == 8, f"Rank has {len(row)} squares: {rank_str}"
        grid.append(row)
    return grid


def crop_board_squares(image: np.ndarray) -> list[list[np.ndarray]]:
    """Crop 64 squares from a 400×400 chess-positions board image.

    Returns squares[row][col] as BGR numpy arrays (50×50×3).
    """
    squares: list[list[np.ndarray]] = []
    for r in range(8):
        row: list[np.ndarray] = []
        for c in range(8):
            y1 = r * SQ_SIZE
            x1 = c * SQ_SIZE
            row.append(image[y1 : y1 + SQ_SIZE, x1 : x1 + SQ_SIZE])
        squares.append(row)
    return squares


def sample_chess_positions_squares(
    data_dir: str | Path,
    max_per_class: int = 1500,
    size: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample balanced per-class squares from chess-positions boards.

    Scans boards until all 13 classes have *max_per_class* samples (or data
    is exhausted).  Empty squares are subsampled to match piece classes.

    Returns ``(images, labels)`` where images are ``(N, size, size, 3)``
    uint8 BGR and labels are ``(N,)`` int64.
    """
    data_dir = Path(data_dir)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith((".jpeg", ".jpg", ".png")))
    rng = random.Random(seed)
    rng.shuffle(files)

    per_class: dict[int, list[np.ndarray]] = {c: [] for c in range(NUM_CLASSES)}
    target = max_per_class

    scanned = 0
    for fname in files:
        # Stop early if all classes are full
        if all(len(per_class[c]) >= target for c in range(NUM_CLASSES)):
            break

        path = data_dir / fname
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if image.shape[:2] != (BOARD_SIZE, BOARD_SIZE):
            continue

        fen = parse_fen_from_filename(fname)
        try:
            labels_grid = fen_to_square_labels(fen)
        except (AssertionError, Exception):
            continue

        squares = crop_board_squares(image)
        for r in range(8):
            for c in range(8):
                cls = labels_grid[r][c]
                if len(per_class[cls]) >= target:
                    continue
                sq = squares[r][c]
                if size != SQ_SIZE:
                    sq = cv2.resize(sq, (size, size))
                per_class[cls].append(sq)

        scanned += 1
        if scanned % 1000 == 0:
            counts = {CLASS_NAMES[c]: len(per_class[c]) for c in range(NUM_CLASSES)}
            logger.info("Scanned %d boards, counts: %s", scanned, counts)

    # Build arrays
    images_list: list[np.ndarray] = []
    labels_list: list[int] = []
    for cls in range(NUM_CLASSES):
        for img in per_class[cls]:
            images_list.append(img)
            labels_list.append(cls)

    images_arr = np.array(images_list, dtype=np.uint8)
    labels_arr = np.array(labels_list, dtype=np.int64)

    # Shuffle
    np_rng = np.random.RandomState(seed)
    perm = np_rng.permutation(len(images_arr))
    images_arr, labels_arr = images_arr[perm], labels_arr[perm]

    total = len(images_arr)
    logger.info(
        "Sampled %d squares from %d boards (%d per class target)",
        total, scanned, target,
    )
    for cls in range(NUM_CLASSES):
        logger.info("  %6s: %d", CLASS_NAMES[cls], len(per_class[cls]))

    return images_arr, labels_arr
