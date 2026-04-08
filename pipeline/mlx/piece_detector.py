"""Piece detection combining the overlay classifier and a VLM fallback.

Detection priority:
1. Grid detection + DINOv2 PieceClassifier (proven for 2D overlays)
2. VLM direct board reading (fallback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from pipeline.analysis.config import VideoAnalysisConfig

logger = logging.getLogger(__name__)

@dataclass
class BoardState:
    """Board state detected from a single frame."""

    fen: str  # Piece placement FEN
    confidence: float
    method: str  # "grid_classifier", "vlm_direct"
    piece_map: dict[str, str] = field(default_factory=dict)


def _detect_with_grid_classifier(
    board_crop_bgr: np.ndarray,
    device: str,
) -> BoardState | None:
    """Detect pieces using grid detector + DINOv2 piece classifier.

    Args:
        board_crop_bgr: (H, W, 3) BGR cropped board image.

    Returns:
        BoardState with FEN, or None if grid detection failed.
    """
    from pipeline.overlay.piece_classifier import read_fen_from_frame

    try:
        fen = read_fen_from_frame(board_crop_bgr, device=device)
        if fen is None:
            return None

        return BoardState(
            fen=fen,
            confidence=0.9,
            method="grid_classifier",
        )
    except Exception as e:
        logger.debug("Grid classifier failed: %s", e)
        return None


def _detect_with_vlm(
    board_crop_rgb: np.ndarray,
    config: VideoAnalysisConfig,
) -> BoardState | None:
    """Use VLM to directly read the board position.

    Args:
        board_crop_rgb: (H, W, 3) RGB board image.
        config: Pipeline configuration.

    Returns:
        BoardState with FEN from VLM, or None.
    """
    from pipeline.mlx.vlm_analyzer import read_board_position

    fen = read_board_position(board_crop_rgb, config)
    if fen is None:
        return None

    return BoardState(
        fen=fen,
        confidence=0.6,
        method="vlm_direct",
    )


def detect_pieces(
    board_crop: np.ndarray,
    config: VideoAnalysisConfig,
    segmenter_method: str = "",
) -> BoardState | None:
    """Detect pieces on a board crop and return FEN.

    The crop may be BGR (from the runtime overlay detector) or RGB (from SAM 3).
    The segmenter_method hint tells us the format.

    Args:
        board_crop: (H, W, 3) cropped board image.
        config: Pipeline configuration.
        segmenter_method: How the crop was produced ("overlay_runtime" and
            "contour" produce BGR crops; "sam3" produces RGB).

    Returns:
        BoardState with FEN, or None if all methods fail.
    """
    import cv2

    # Determine color format.  Overlay scanner and contour fallback
    # both crop from a BGR frame, so the crop is already BGR.
    # SAM 3 crops from an RGB frame.
    if segmenter_method == "sam3":
        board_bgr = cv2.cvtColor(board_crop, cv2.COLOR_RGB2BGR)
        board_rgb = board_crop
    else:
        # overlay_runtime and contour both produce BGR crops
        board_bgr = board_crop
        board_rgb = cv2.cvtColor(board_crop, cv2.COLOR_BGR2RGB)

    # Path 1: Grid + PieceClassifier (fast, proven for overlays)
    if config.use_piece_classifier:
        result = _detect_with_grid_classifier(board_bgr, config.device)
        if result is not None:
            return result

    # Path 2: VLM direct board reading
    result = _detect_with_vlm(board_rgb, config)
    if result is not None:
        return result

    logger.warning("All piece detection methods failed")
    return None
