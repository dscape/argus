"""Board segmentation helpers for analysis fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from pipeline.analysis.config import VideoAnalysisConfig

logger = logging.getLogger(__name__)

_sam_model: Any = None
_sam_processor: Any = None
_sam_loaded = False


@dataclass
class BoardSegment:
    """Segmented board crop from a full frame."""

    bbox: tuple[int, int, int, int]
    cropped_board: np.ndarray
    method: str


def _detect_with_overlay_scanner(frame_bgr: np.ndarray) -> BoardSegment | None:
    from pipeline.overlay.scanner import detect_overlay_runtime

    detection = detect_overlay_runtime(frame_bgr)
    if not detection.found or detection.bbox is None:
        return None

    x, y, w, h = detection.bbox
    frame_h, frame_w = frame_bgr.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    if w < 50 or h < 50:
        return None

    logger.debug("Overlay scanner found board: (%d,%d,%d,%d)", x, y, w, h)
    return BoardSegment(
        bbox=(x, y, w, h),
        cropped_board=frame_bgr[y : y + h, x : x + w].copy(),
        method="overlay_runtime",
    )


def _load_sam(config: VideoAnalysisConfig) -> None:
    global _sam_model, _sam_processor, _sam_loaded

    if _sam_loaded:
        return

    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        logger.debug("Text-prompted segmentation backend unavailable")
        return

    logger.info("Loading segmentation model")
    _sam_model = build_sam3_image_model()
    _sam_processor = Sam3Processor(_sam_model, confidence_threshold=config.sam_confidence_threshold)
    _sam_loaded = True
    logger.info("Segmentation model loaded")


def _detect_with_sam3(frame_rgb: np.ndarray, config: VideoAnalysisConfig) -> BoardSegment | None:
    _load_sam(config)
    if not _sam_loaded:
        return None

    state = _sam_processor.set_image(Image.fromarray(frame_rgb))
    state = _sam_processor.set_text_prompt("chess board", state)

    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or len(masks) == 0:
        return None

    scores_arr = np.array(scores)
    best_idx = int(np.argmax(scores_arr))
    confidence = float(scores_arr[best_idx])
    if confidence < config.sam_confidence_threshold:
        return None

    mask = masks[best_idx]
    mask_np = mask.numpy().astype(bool) if hasattr(mask, "numpy") else np.array(mask, dtype=bool)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return BoardSegment(
        bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
        cropped_board=frame_rgb[y_min : y_max + 1, x_min : x_max + 1].copy(),
        method="sam3",
    )


def _detect_with_contours(frame_bgr: np.ndarray) -> BoardSegment | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = frame_bgr.shape[:2]
    min_area = (frame_h * frame_w) * 0.02
    best_rect: tuple[int, int, int, int] | None = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if 4 <= len(approx) <= 6 and area > best_area:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h > 0 else 0
            if 0.7 <= aspect <= 1.4:
                best_area = area
                best_rect = (x, y, w, h)

    if best_rect is None:
        return None

    x, y, w, h = best_rect
    return BoardSegment(
        bbox=best_rect,
        cropped_board=frame_bgr[y : y + h, x : x + w].copy(),
        method="contour",
    )


def segment_board(frame: np.ndarray, config: VideoAnalysisConfig) -> BoardSegment | None:
    """Segment the chess board from an RGB video frame."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    result = _detect_with_overlay_scanner(frame_bgr)
    if result is not None:
        return result

    result = _detect_with_sam3(frame, config)
    if result is not None:
        return result

    return _detect_with_contours(frame_bgr)


def unload_model() -> None:
    """Free segmentation model memory."""
    global _sam_model, _sam_processor, _sam_loaded
    _sam_model = None
    _sam_processor = None
    _sam_loaded = False
    logger.info("Segmentation model unloaded")
