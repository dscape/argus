"""Board-state reading with overlay and VLM fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.analysis.vlm import read_board_position

logger = logging.getLogger(__name__)


@dataclass
class BoardState:
    """Board state detected from a board crop."""

    fen: str
    method: str


def _detect_with_grid_classifier(board_crop_bgr: np.ndarray, device: str) -> BoardState | None:
    from pipeline.overlay.piece_classifier import read_fen_from_frame

    try:
        fen = read_fen_from_frame(board_crop_bgr, device=device)
    except Exception as exc:  # pragma: no cover - defensive logging around model runtime
        logger.debug("Grid classifier failed: %s", exc)
        return None

    if fen is None:
        return None
    return BoardState(fen=fen, method="grid_classifier")


def _detect_with_vlm(board_crop_rgb: np.ndarray, config: VideoAnalysisConfig) -> BoardState | None:
    fen = read_board_position(board_crop_rgb, config)
    if fen is None:
        return None
    return BoardState(fen=fen, method="vlm_direct")


def detect_pieces(
    board_crop: np.ndarray,
    config: VideoAnalysisConfig,
    segmenter_method: str = "",
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
        result = _detect_with_grid_classifier(board_bgr, config.device)
        if result is not None:
            return result

    result = _detect_with_vlm(board_rgb, config)
    if result is not None:
        return result

    logger.warning("All board-reading methods failed")
    return None
