"""Board-state reading with physical square-classifier and VLM fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.analysis.vlm import read_board_position

logger = logging.getLogger(__name__)


@dataclass
class BoardState:
    """Board state detected from a board crop."""

    fen: str
    method: str


def _detect_with_square_classifier(
    board_crop_bgr: np.ndarray,
    device: str,
    *,
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
    sequence_reader: Any | None = None,
) -> BoardState | None:
    from pipeline.physical.square_classifier import read_fen_from_frame

    try:
        if sequence_reader is None:
            fen = read_fen_from_frame(board_crop_bgr, corners=corners, device=device)
        elif corners is None:
            fen = sequence_reader.read_fen_from_frame(board_crop_bgr)
        else:
            fen = sequence_reader.read_fen_from_frame(board_crop_bgr, corners=corners)
    except Exception as exc:  # pragma: no cover - defensive logging around model runtime
        logger.debug("Physical square classifier failed: %s", exc)
        return None

    if fen is None:
        return None
    return BoardState(fen=fen, method="physical_square_classifier")


def _detect_with_vlm(board_crop_rgb: np.ndarray, config: VideoAnalysisConfig) -> BoardState | None:
    fen = read_board_position(board_crop_rgb, config)
    if fen is None:
        return None
    return BoardState(fen=fen, method="vlm_direct")


def detect_pieces(
    board_crop: np.ndarray,
    config: VideoAnalysisConfig,
    segmenter_method: str = "",
    *,
    board_corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
    physical_sequence_reader: Any | None = None,
) -> BoardState | None:
    """Detect pieces on a board crop and return a FEN string."""
    import cv2

    if segmenter_method == "sam3":
        board_bgr = cv2.cvtColor(board_crop, cv2.COLOR_RGB2BGR)
        board_rgb = board_crop
    else:
        board_bgr = board_crop
        board_rgb = cv2.cvtColor(board_crop, cv2.COLOR_BGR2RGB)

    if config.use_piece_classifier:
        result = _detect_with_square_classifier(
            board_bgr,
            config.device,
            corners=board_corners,
            sequence_reader=physical_sequence_reader,
        )
        if result is not None:
            return result

    if board_corners is not None:
        board_rgb = _crop_rgb_to_board_corners(board_rgb, board_corners)

    result = _detect_with_vlm(board_rgb, config)
    if result is not None:
        return result

    logger.warning("All board-reading methods failed")
    return None


def _crop_rgb_to_board_corners(
    image_rgb: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {points.shape}")
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    extent = np.maximum(max_xy - min_xy, 1.0)
    margin = extent * 0.15
    height, width = image_rgb.shape[:2]
    x1 = max(0, int(np.floor(min_xy[0] - margin[0])))
    y1 = max(0, int(np.floor(min_xy[1] - margin[1])))
    x2 = min(width, int(np.ceil(max_xy[0] + margin[0])))
    y2 = min(height, int(np.ceil(max_xy[1] + margin[1])))
    return image_rgb[y1:y2, x1:x2].copy()
