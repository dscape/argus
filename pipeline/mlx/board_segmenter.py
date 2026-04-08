"""Board segmentation for the optional MLX analysis fallback.

Detection priority:
1. Runtime overlay detector — default path for rendered 2D overlays.
2. SAM 3 text-prompted segmentation — for 3D/OTB boards without overlays.
3. OpenCV contour fallback — last resort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from pipeline.analysis.config import VideoAnalysisConfig

logger = logging.getLogger(__name__)

# Lazy-loaded SAM 3 model cache
_sam_model: Any = None
_sam_processor: Any = None
_sam_loaded: bool = False


@dataclass
class BoardSegment:
    """Result of board segmentation."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixel coordinates
    mask: np.ndarray  # (H, W) boolean mask
    confidence: float
    cropped_board: np.ndarray  # (h, w, 3) cropped board region
    method: str  # "overlay_runtime", "sam3", "contour"


def _detect_with_overlay_scanner(frame_bgr: np.ndarray) -> BoardSegment | None:
    """Use the runtime overlay detector to find a rendered 2D board."""
    from pipeline.overlay.scanner import detect_overlay_runtime

    detection = detect_overlay_runtime(frame_bgr)

    if not detection.found or detection.bbox is None:
        return None

    x, y, w, h = detection.bbox
    frame_h, frame_w = frame_bgr.shape[:2]

    # Clamp bbox to frame bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    if w < 50 or h < 50:
        return None

    mask = np.zeros((frame_h, frame_w), dtype=bool)
    mask[y : y + h, x : x + w] = True
    cropped = frame_bgr[y : y + h, x : x + w].copy()

    logger.debug(
        "Overlay scanner found board: (%d,%d,%d,%d) score=%.2f",
        x, y, w, h, detection.score,
    )

    return BoardSegment(
        bbox=(x, y, w, h),
        mask=mask,
        confidence=detection.score,
        cropped_board=cropped,
        method="overlay_runtime",
    )


def _load_sam(config: VideoAnalysisConfig) -> None:
    """Lazy-load SAM 3 model."""
    global _sam_model, _sam_processor, _sam_loaded

    if _sam_loaded:
        return

    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        logger.info("Loading SAM 3 model")
        _sam_model = build_sam3_image_model()
        _sam_processor = Sam3Processor(
            _sam_model, confidence_threshold=config.sam_confidence_threshold
        )
        _sam_loaded = True
        logger.info("SAM 3 model loaded")
    except ImportError:
        logger.debug("sam3 not available, will use contour fallback for non-overlay boards")


def _detect_with_sam3(frame_rgb: np.ndarray, config: VideoAnalysisConfig) -> BoardSegment | None:
    """Use SAM 3 text-prompted segmentation."""
    _load_sam(config)

    if not _sam_loaded:
        return None

    image = Image.fromarray(frame_rgb)
    h, w = frame_rgb.shape[:2]

    state = _sam_processor.set_image(image)
    state = _sam_processor.set_text_prompt("chess board", state)

    masks = state.get("masks")
    scores = state.get("scores")

    if masks is None or len(masks) == 0:
        return None

    scores_arr = np.array(scores)
    best_idx = int(np.argmax(scores_arr))
    mask = masks[best_idx]
    confidence = float(scores_arr[best_idx])

    if confidence < config.sam_confidence_threshold:
        return None

    if hasattr(mask, "numpy"):
        mask_np = mask.numpy().astype(bool)
    else:
        mask_np = np.array(mask, dtype=bool)

    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    cropped = frame_rgb[y_min : y_max + 1, x_min : x_max + 1].copy()

    return BoardSegment(
        bbox=bbox,
        mask=mask_np,
        confidence=confidence,
        cropped_board=cropped,
        method="sam3",
    )


def _detect_with_contours(frame_bgr: np.ndarray) -> BoardSegment | None:
    """Fallback: find the largest roughly-square contour."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame_bgr.shape[:2]
    min_area = (h * w) * 0.02
    best_rect: tuple[int, int, int, int] | None = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if 4 <= len(approx) <= 6 and area > best_area:
            bx, by, bw, bh = cv2.boundingRect(approx)
            aspect = bw / bh if bh > 0 else 0
            if 0.7 <= aspect <= 1.4:
                best_area = area
                best_rect = (bx, by, bw, bh)

    if best_rect is None:
        return None

    bx, by, bw, bh = best_rect
    mask = np.zeros((h, w), dtype=bool)
    mask[by : by + bh, bx : bx + bw] = True
    cropped = frame_bgr[by : by + bh, bx : bx + bw].copy()

    return BoardSegment(
        bbox=best_rect,
        mask=mask,
        confidence=0.3,
        cropped_board=cropped,
        method="contour",
    )


def segment_board(
    frame: np.ndarray,
    config: VideoAnalysisConfig,
) -> BoardSegment | None:
    """Segment the chess board from a video frame.

    Tries methods in order of reliability:
    1. Overlay scanner (fast, proven for broadcast overlays)
    2. SAM 3 text-prompted segmentation (for 3D/OTB boards)
    3. OpenCV contour fallback

    Args:
        frame: (H, W, 3) RGB image.
        config: Pipeline configuration.

    Returns:
        BoardSegment with the board region, or None if no board found.
    """
    # The overlay scanner expects BGR (OpenCV convention).
    # The shared analysis pipeline passes RGB frames from PyAV, so convert here.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 1. Runtime overlay detector — best for broadcast chess videos
    result = _detect_with_overlay_scanner(frame_bgr)
    if result is not None:
        return result

    # 2. SAM 3 — for 3D boards (needs RGB input)
    result = _detect_with_sam3(frame, config)
    if result is not None:
        return result

    # 3. Contour fallback
    result = _detect_with_contours(frame_bgr)
    if result is not None:
        return result

    return None


def unload_model() -> None:
    """Free SAM 3 model memory."""
    global _sam_model, _sam_processor, _sam_loaded
    _sam_model = None
    _sam_processor = None
    _sam_loaded = False
    logger.info("SAM 3 model unloaded")
