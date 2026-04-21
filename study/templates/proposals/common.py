from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pipeline.physical.piece_projection import board_to_image_homography


@dataclass(frozen=True)
class ProposalFrame:
    image_bgr: np.ndarray
    corners: tuple[tuple[float, float], ...]
    frame_id: str | None = None


@dataclass(frozen=True)
class SquareCropProposal:
    square: str
    crop_bgr: np.ndarray
    score: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    mask: np.ndarray | None = None


def board_point_to_square_name(
    corners: tuple[tuple[float, float], ...],
    point_xy: tuple[float, float],
) -> str | None:
    homography = board_to_image_homography(corners)
    inverse_homography = np.linalg.inv(homography)
    point = np.asarray([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    board_point = cv2.perspectiveTransform(point, inverse_homography)[0, 0]
    board_x = float(board_point[0])
    board_y = float(board_point[1])
    if board_x < 0.0 or board_x >= 8.0 or board_y < 0.0 or board_y >= 8.0:
        return None
    row_index = int(np.floor(board_y))
    col_index = int(np.floor(board_x))
    rank = 8 - row_index
    file_name = chr(ord("a") + col_index)
    return f"{file_name}{rank}"


__all__ = [
    "ProposalFrame",
    "SquareCropProposal",
    "board_point_to_square_name",
]
