"""Utilities for labeled real overlay board crops.

Real board-crop filenames encode piece-placement FEN:
- ``r{clip_id}_{fen-hyphenated}.jpg``
- ``f_{video_id}_{frame_name}_{fen-hyphenated}.jpg``

These images are already cropped close to the board but can have arbitrary
resolutions, minor borders, or UI chrome. We recover a grid with ``detect_grid``
and fall back to a uniform 8×8 split when line detection is unavailable.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import overload

import cv2
import numpy as np

from pipeline.overlay.grid_detector import GridResult, detect_grid
from pipeline.shared import NUM_SQUARE_CLASSES, fen_to_square_labels

logger = logging.getLogger(__name__)

_KNOWN_FRAME_NAMES = ("thumb_hires", "thumb_sd", "thumb", "25pct", "50pct", "75pct")
_IMAGE_EXTS = (".jpeg", ".jpg", ".png")
_LABEL_FIXUPS = {
    "f_lQXos0du0bg_25pct_2p1bppp-p7-3pR3-p2P4-1P6-P1P2PPP-RNBQ2K1-8": (
        "6k1/1p2pp2/2q4p/2p5/2P5/7P/1PQ1PP2/4R1K1"
    ),
}


@overload
def parse_real_board_fen(filename: str) -> str: ...


@overload
def parse_real_board_fen(filename: Path) -> str: ...


def parse_real_board_fen(filename: str | Path) -> str:
    """Extract a piece-placement FEN string from a real board-crop filename."""
    stem = Path(filename).stem
    fixed = _LABEL_FIXUPS.get(stem)
    if fixed is not None:
        return fixed
    if stem.startswith("f_"):
        return _parse_frame_board_fen(stem)
    if stem.startswith("r") and "_" in stem:
        return stem.split("_", 1)[1].replace("-", "/")
    return stem.replace("-", "/")


def sample_real_board_squares(
    data_dir: str | Path,
    max_per_class: int | dict[int, int] | None = None,
    size: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample labeled square crops from real board images.

    If ``max_per_class`` is ``None``, all squares are returned. Otherwise the
    sampler stops once every class reaches its target count.
    """
    data_dir = Path(data_dir)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(_IMAGE_EXTS))
    rng = random.Random(seed)
    rng.shuffle(files)

    if max_per_class is None:
        targets: dict[int, int] | None = None
        per_class = None
    elif isinstance(max_per_class, int):
        targets = {class_index: max_per_class for class_index in range(NUM_SQUARE_CLASSES)}
        per_class = {class_index: [] for class_index in range(NUM_SQUARE_CLASSES)}
    else:
        targets = {
            class_index: int(max_per_class.get(class_index, 0))
            for class_index in range(NUM_SQUARE_CLASSES)
        }
        per_class = {class_index: [] for class_index in range(NUM_SQUARE_CLASSES)}

    images_list: list[np.ndarray] = []
    labels_list: list[int] = []
    scanned = 0

    for fname in files:
        if targets is not None and per_class is not None:
            if all(len(per_class[c]) >= targets[c] for c in range(NUM_SQUARE_CLASSES)):
                break

        path = data_dir / fname
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        try:
            labels_grid = fen_to_square_labels(parse_real_board_fen(fname))
        except (AssertionError, ValueError) as exc:
            logger.warning("Skipping malformed real board label %s: %s", fname, exc)
            continue

        grid = detect_grid(image) or _uniform_grid_for_image(image)
        squares = grid.crop_squares(image)

        for row in range(8):
            for col in range(8):
                label = labels_grid[row][col]
                if targets is not None and per_class is not None:
                    if len(per_class[label]) >= targets[label]:
                        continue
                square = squares[row][col]
                if size != square.shape[0] or size != square.shape[1]:
                    square = cv2.resize(square, (size, size))
                if targets is None:
                    images_list.append(square)
                    labels_list.append(label)
                else:
                    assert per_class is not None
                    per_class[label].append(square)

        scanned += 1

    if targets is not None and per_class is not None:
        for class_index in range(NUM_SQUARE_CLASSES):
            for image in per_class[class_index]:
                images_list.append(image)
                labels_list.append(class_index)

    images_arr = np.array(images_list, dtype=np.uint8)
    labels_arr = np.array(labels_list, dtype=np.int64)
    perm = np.random.RandomState(seed).permutation(len(images_arr))
    images_arr, labels_arr = images_arr[perm], labels_arr[perm]

    logger.info("Sampled %d real-board squares from %d boards", len(images_arr), scanned)
    return images_arr, labels_arr


def _parse_frame_board_fen(stem: str) -> str:
    for frame_name in _KNOWN_FRAME_NAMES:
        marker = f"_{frame_name}_"
        idx = stem.find(marker)
        if idx > 0:
            fen_hyphenated = stem[idx + len(marker) :]
            return fen_hyphenated.replace("-", "/")
    raise ValueError(f"Cannot parse real frame board filename: {stem}")


def _uniform_grid_for_image(image: np.ndarray) -> GridResult:
    h, w = image.shape[:2]
    sq_w = w / 8
    sq_h = h / 8
    sq_size = int(round((sq_w + sq_h) / 2))
    return GridResult(
        v_lines=[int(round(col * sq_w)) for col in range(9)],
        h_lines=[int(round(row * sq_h)) for row in range(9)],
        sq_size=sq_size,
    )
