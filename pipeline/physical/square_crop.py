"""Per-square crops for occupancy and piece classifiers.

Two extraction modes share this module:

- ``extract_occupancy_crop`` / ``extract_piece_crop`` (original chesscog-geometry
  fallback). Board-coord extension with fixed constants; doesn't adapt to
  camera angle and produces mostly-background crops for near-top-down cameras.
- ``extract_projected_piece_crop`` (re-exported from ``piece_projection``).
  Recovers the camera pose from the 4 board corners via ``cv2.solvePnP`` and
  projects a 1x1x``piece_height`` 3D piece box through ``K[R|t]``. Adapts per
  frame to the actual camera setup.

The projected extractor is the default in new datasets; the chesscog fallback
stays here so older tests and ad-hoc debugging tools keep working.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.physical.oblique_square_context import board_to_image_transform

DEFAULT_OCCUPANCY_CROP_SIZE = 112
DEFAULT_PIECE_CROP_SIZE = 224

OCCUPANCY_PAD = 0.5

PIECE_TOP_MIN = 1.0
PIECE_TOP_MAX = 3.0
PIECE_SIDE_MIN = 0.25
PIECE_SIDE_MAX = 1.0

_PIECE_CANVAS_SQUARES = 4.0


def extract_occupancy_crop(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    row: int,
    col: int,
    output_size: int = DEFAULT_OCCUPANCY_CROP_SIZE,
) -> np.ndarray:
    """Return one symmetric-padded square crop for the occupancy classifier."""
    _validate_row_col(row, col)
    pad = OCCUPANCY_PAD
    board_quad = np.array(
        [
            [col - pad, row - pad],
            [col + 1 + pad, row - pad],
            [col + 1 + pad, row + 1 + pad],
            [col - pad, row + 1 + pad],
        ],
        dtype=np.float32,
    )
    image_quad = _project_board_quad_to_image(board_quad, corners)
    destination = np.array(
        [
            [0.0, 0.0],
            [float(output_size), 0.0],
            [float(output_size), float(output_size)],
            [0.0, float(output_size)],
        ],
        dtype=np.float32,
    )
    warp = cv2.getPerspectiveTransform(image_quad, destination)
    return cv2.warpPerspective(image_bgr, warp, (output_size, output_size))


def extract_piece_crop(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    row: int,
    col: int,
    output_size: int = DEFAULT_PIECE_CROP_SIZE,
) -> np.ndarray:
    """Return one chesscog-style piece crop, mirrored if ``col < 4``.

    The canvas is laid out as a ``_PIECE_CANVAS_SQUARES`` × ``_PIECE_CANVAS_SQUARES``
    virtual grid (``ss_canvas = output_size / 4``). The target square always
    lands at the bottom-left tile. Top/side extensions (up to 3 squares up and
    1 square sideways) fill the remaining canvas space. Zero pixels pad the
    top-right region when the extensions do not reach that far.
    """
    _validate_row_col(row, col)
    top_ext = _piece_top_extension(row)
    left_ext, right_ext = _piece_side_extensions(col)

    if col < 4:
        side_ext = left_ext
        board_quad = np.array(
            [
                [col - side_ext, row - top_ext],
                [col + 1, row - top_ext],
                [col + 1, row + 1],
                [col - side_ext, row + 1],
            ],
            dtype=np.float32,
        )
    else:
        side_ext = right_ext
        board_quad = np.array(
            [
                [col, row - top_ext],
                [col + 1 + side_ext, row - top_ext],
                [col + 1 + side_ext, row + 1],
                [col, row + 1],
            ],
            dtype=np.float32,
        )

    image_quad = _project_board_quad_to_image(board_quad, corners)

    ss_canvas = output_size / _PIECE_CANVAS_SQUARES
    local_w = max(1, int(round((1.0 + side_ext) * ss_canvas)))
    local_h = max(1, int(round((1.0 + top_ext) * ss_canvas)))
    local_dst = np.array(
        [
            [0.0, 0.0],
            [float(local_w), 0.0],
            [float(local_w), float(local_h)],
            [0.0, float(local_h)],
        ],
        dtype=np.float32,
    )
    warp = cv2.getPerspectiveTransform(image_quad, local_dst)
    local = cv2.warpPerspective(image_bgr, warp, (local_w, local_h))

    if col < 4:
        local = cv2.flip(local, 1)

    canvas = np.zeros((output_size, output_size, image_bgr.shape[2]), dtype=image_bgr.dtype)
    y_offset = output_size - local_h
    canvas[y_offset : y_offset + local_h, 0:local_w] = local
    return canvas


def extract_all_occupancy_crops(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    output_size: int = DEFAULT_OCCUPANCY_CROP_SIZE,
) -> list[np.ndarray]:
    """Return 64 occupancy crops in row-major order (a8, b8, ..., h1)."""
    return [
        extract_occupancy_crop(
            image_bgr,
            corners,
            row=index // 8,
            col=index % 8,
            output_size=output_size,
        )
        for index in range(64)
    ]


def extract_all_piece_crops(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    output_size: int = DEFAULT_PIECE_CROP_SIZE,
) -> list[np.ndarray]:
    """Return 64 piece crops in row-major order (a8, b8, ..., h1)."""
    return [
        extract_piece_crop(
            image_bgr,
            corners,
            row=index // 8,
            col=index % 8,
            output_size=output_size,
        )
        for index in range(64)
    ]


def _piece_top_extension(row: int) -> float:
    depth = (7 - row) / 7.0
    return PIECE_TOP_MIN + (PIECE_TOP_MAX - PIECE_TOP_MIN) * depth


def _piece_side_extensions(col: int) -> tuple[float, float]:
    if col < 4:
        depth = (3 - col) / 3.0
        return PIECE_SIDE_MIN + (PIECE_SIDE_MAX - PIECE_SIDE_MIN) * depth, 0.0
    depth = (col - 4) / 3.0
    return 0.0, PIECE_SIDE_MIN + (PIECE_SIDE_MAX - PIECE_SIDE_MIN) * depth


def _project_board_quad_to_image(
    board_quad: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
) -> np.ndarray:
    transform = board_to_image_transform(corners)
    projected = cv2.perspectiveTransform(board_quad.reshape(1, 4, 2), transform)
    return projected.reshape(4, 2).astype(np.float32)


def _validate_row_col(row: int, col: int) -> None:
    if not 0 <= row <= 7 or not 0 <= col <= 7:
        raise ValueError(f"row and col must be in [0, 7], got row={row}, col={col}")
