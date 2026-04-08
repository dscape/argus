"""Detect both a 2D overlay and an OTB chessboard in video frames."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.overlay.scanner import (
    OverlayDetection,
    extract_frames_from_video,
    runtime_overlay_check,
)

logger = logging.getLogger(__name__)

_OTB_YOLO_WEIGHTS_OVERRIDE_ENV = "ARGUS_OTB_YOLO_WEIGHTS"
_OTB_YOLO_CONF_ENV = "ARGUS_OTB_YOLO_CONF"
_OTB_YOLO_IMGSZ_ENV = "ARGUS_OTB_YOLO_IMGSZ"
_OTB_YOLO_DEVICE_ENV = "ARGUS_OTB_YOLO_DEVICE"
_MIN_VISIBLE_NON_OVERLAY_FRACTION = 0.20


@dataclass
class OTBDetection:
    """Result of OTB-board detection."""

    found: bool
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    frame_resolution: tuple[int, int] | None = None


@dataclass
class ScreeningResult:
    """Combined result of overlay + OTB detection."""

    has_overlay: bool
    has_otb: bool
    overlay_bbox: tuple[int, int, int, int] | None = None
    overlay_score: float = 0.0
    otb_confidence: float = 0.0
    approved: bool = False


def detect_otb_region(
    frame: np.ndarray,
    overlay_bbox: tuple[int, int, int, int],
) -> OTBDetection:
    """Detect the physical OTB board while ignoring the digital overlay."""
    h, w = frame.shape[:2]
    ox, oy, ow, oh = overlay_bbox
    visible_fraction = 1.0 - ((ow * oh) / max(h * w, 1))
    if visible_fraction < _MIN_VISIBLE_NON_OVERLAY_FRACTION:
        return OTBDetection(found=False, frame_resolution=(w, h))

    masked = frame.copy()
    masked[oy : oy + oh, ox : ox + ow] = 0

    detection = _run_otb_yolo_detector(masked)
    if not detection.found or detection.bbox is None:
        return detection

    if _bbox_overlap_fraction(detection.bbox, overlay_bbox) > 0.05:
        return OTBDetection(found=False, frame_resolution=detection.frame_resolution)

    return detection


def screen_video(video_url_or_path: str) -> ScreeningResult:
    """Screen a video for both overlay and OTB board presence."""
    frame_paths = extract_frames_from_video(video_url_or_path)

    if not frame_paths:
        logger.warning("Could not extract frames from %s", video_url_or_path)
        return ScreeningResult(has_overlay=False, has_otb=False)

    best_overlay = OverlayDetection(found=False)
    best_otb = OTBDetection(found=False)
    best_overlay_bbox = None

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        overlay_det = runtime_overlay_check(frame)
        if not overlay_det.found or overlay_det.bbox is None:
            continue

        if overlay_det.score > best_overlay.score:
            best_overlay = overlay_det

        otb_det = detect_otb_region(frame, overlay_det.bbox)
        if otb_det.confidence > best_otb.confidence:
            best_otb = otb_det
            best_overlay_bbox = overlay_det.bbox

    for path in frame_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    has_overlay = best_overlay.found
    has_otb = best_otb.found
    approved = has_overlay and has_otb

    return ScreeningResult(
        has_overlay=has_overlay,
        has_otb=has_otb,
        overlay_bbox=best_overlay_bbox,
        overlay_score=best_overlay.score,
        otb_confidence=best_otb.confidence,
        approved=approved,
    )


def overlay_bbox_to_json(bbox: tuple[int, int, int, int] | None) -> str | None:
    """Convert overlay bbox tuple to JSON string for DB storage."""
    if bbox is None:
        return None
    x, y, w, h = bbox
    return json.dumps({"x": x, "y": y, "w": w, "h": h})


def _run_otb_yolo_detector(frame: np.ndarray) -> OTBDetection:
    """Run the default OTB-board YOLO detector with optional env overrides."""
    from pipeline.screen.otb_yolo_detector import DEFAULT_WEIGHTS_PATH, detect_otb_yolo

    weights_override = os.getenv(_OTB_YOLO_WEIGHTS_OVERRIDE_ENV, "").strip()
    weights_path = weights_override or str(DEFAULT_WEIGHTS_PATH)
    conf = float(os.getenv(_OTB_YOLO_CONF_ENV, "0.20"))
    imgsz = int(os.getenv(_OTB_YOLO_IMGSZ_ENV, "640"))
    device = os.getenv(_OTB_YOLO_DEVICE_ENV, "auto")

    return detect_otb_yolo(
        frame,
        weights_path=weights_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
    )


def _bbox_overlap_fraction(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = float((x2 - x1) * (y2 - y1))
    area_a = float(max(aw * ah, 1))
    return intersection / area_a
