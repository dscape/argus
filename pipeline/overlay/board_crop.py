"""Board-grid localization inside already-cropped overlay regions.

Calibrated overlay crops are sometimes slightly larger than the board itself and
include a border around the 8x8 area. In those cases ``detect_grid(...)`` can
fail and its uniform fallback incorrectly splits the *entire* crop into 64
squares. The square classifier then sees shifted edge squares (for example a8
contains border + rook instead of just the rook square), which looks like a
piece-classifier failure but is really a geometry failure.

This module first tries strict grid detection without the uniform fallback. If
that fails, it searches for the best square board sub-region inside the crop and
returns a uniform 8x8 grid *within that localized square*.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Final

import cv2
import numpy as np

from pipeline.overlay.grid_detector import GridResult, detect_grid
from pipeline.overlay.scanner import (
    compute_alternation_strength,
    compute_axis_aligned_periodicity,
    compute_grid_regularity,
)

_MIN_BOARD_SIDE: Final[int] = 64


def find_board_grid_in_crop(overlay_crop: np.ndarray) -> GridResult | None:
    """Return the best 8x8 grid inside an already-cropped overlay region."""
    if overlay_crop.size == 0:
        return None

    strict_grid = detect_grid(overlay_crop, allow_uniform=False)
    if strict_grid is not None:
        return strict_grid

    bbox = _localize_square_board_bbox(overlay_crop, strict_grid)
    if bbox is None:
        return detect_grid(overlay_crop)

    return _uniform_grid_for_bbox(*bbox)


def find_stable_board_grid(
    crop_loader: Callable[[int], np.ndarray | None],
    frame_indices: Iterable[int],
) -> GridResult | None:
    """Pick one stable board grid for a sequence of overlay crops.

    The board geometry inside a calibrated clip is effectively constant, so a
    grid detected on any clean frame can be reused for noisier neighbors.
    """
    fallback_crop: np.ndarray | None = None
    seen: set[int] = set()
    for frame_index in frame_indices:
        if frame_index in seen:
            continue
        seen.add(frame_index)
        crop = crop_loader(frame_index)
        if crop is None or crop.size == 0:
            continue
        strict_grid = detect_grid(crop, allow_uniform=False)
        if strict_grid is not None:
            return strict_grid
        if fallback_crop is None:
            fallback_crop = crop

    if fallback_crop is None:
        return None
    return find_board_grid_in_crop(fallback_crop)


def _localize_square_board_bbox(
    overlay_crop: np.ndarray,
    strict_grid: GridResult | None,
) -> tuple[int, int, int] | None:
    best_bbox: tuple[int, int, int] | None = None
    best_score = -1.0

    for seed_bbox in _initial_board_crop_seeds(overlay_crop, strict_grid):
        bbox = _refine_square_bbox(overlay_crop, seed_bbox)
        x, y, side = bbox
        region = overlay_crop[y : y + side, x : x + side]
        score = _board_alignment_score(region)
        if score > best_score:
            best_bbox = bbox
            best_score = score

    return best_bbox


def _board_alignment_score(image: np.ndarray) -> float:
    regularity = compute_grid_regularity(image)
    frac, contrast = compute_alternation_strength(image)
    periodicity = compute_axis_aligned_periodicity(image)
    return regularity * frac * contrast * periodicity


def _square_bbox_from_grid(
    image_shape: tuple[int, ...],
    grid: GridResult,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    gx1, gx2 = grid.v_lines[0], grid.v_lines[-1]
    gy1, gy2 = grid.h_lines[0], grid.h_lines[-1]
    side = max(gx2 - gx1, gy2 - gy1)
    x = max(0, min(gx1, width - side))
    y = max(0, min(gy1, height - side))
    side = min(side, width - x, height - y)
    return x, y, side, side


def _initial_board_crop_seeds(
    image: np.ndarray,
    grid: GridResult | None,
) -> list[tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    side = min(height, width)
    seeds = {
        (0, 0, side, side),
        (max(0, (width - side) // 2), max(0, (height - side) // 2), side, side),
    }
    if grid is not None:
        seeds.add(_square_bbox_from_grid(image.shape, grid))
    return list(seeds)


def _refine_square_bbox(
    image: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    height, width = image.shape[:2]
    seed_x, seed_y, seed_w, seed_h = seed_bbox
    seed_side = min(seed_w, seed_h, height, width)
    seed_x = max(0, min(seed_x, width - seed_side))
    seed_y = max(0, min(seed_y, height - seed_side))

    cell = max(1, seed_side // 8)
    step = max(2, cell // 6)
    size_slack = max(2, cell // 2)
    side_min = max(_MIN_BOARD_SIDE, seed_side - size_slack)
    side_max = min(min(height, width), seed_side + size_slack)

    best_x = seed_x
    best_y = seed_y
    best_side = seed_side
    best_score = _board_alignment_score(
        image[seed_y : seed_y + seed_side, seed_x : seed_x + seed_side]
    )

    for side in range(side_min, side_max + 1, step):
        min_x = max(0, seed_x - cell)
        max_x = min(width - side, seed_x + cell)
        min_y = max(0, seed_y - cell)
        max_y = min(height - side, seed_y + cell)

        for x in range(min_x, max_x + 1, step):
            for y in range(min_y, max_y + 1, step):
                score = _board_alignment_score(image[y : y + side, x : x + side])
                if score > best_score:
                    best_x = x
                    best_y = y
                    best_side = side
                    best_score = score

    return best_x, best_y, best_side


def _uniform_grid_for_bbox(x: int, y: int, side: int) -> GridResult:
    square_size = side / 8.0
    return GridResult(
        v_lines=[x + int(round(col * square_size)) for col in range(9)],
        h_lines=[y + int(round(row * square_size)) for row in range(9)],
        sq_size=int(round(square_size)),
    )
