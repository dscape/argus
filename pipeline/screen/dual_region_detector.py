"""Detect both a 2D overlay AND real OTB camera footage in video frames.

Extends the overlay scanner to also verify that the video contains real
over-the-board camera footage — not just a full-screen rendered board or
a solid-color background behind the overlay.
"""

import json
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.overlay.scanner import (
    OverlayDetection,
    detect_overlay_in_frame,
    extract_frames_from_video,
)

logger = logging.getLogger(__name__)

# Minimum Laplacian variance for natural textures (wood grain, people, etc.)
# Real camera footage of tournament scenes typically has variance 300-800+.
# Rendered UIs, move lists, and sidebar chrome sit around 50-150.
MIN_LAPLACIAN_VARIANCE = 200.0

# Minimum color standard deviation across the OTB region.
# Real scenes have diverse colors from lighting, skin tones, wood, clothing.
# UI elements and rendered boards have limited palettes (typically std < 25).
MIN_COLOR_STD = 30.0

# Minimum fraction of frame area that must be non-overlay to be considered
# as potentially containing OTB footage. Needs a substantial camera region.
MIN_OTB_AREA_FRACTION = 0.25


@dataclass
class OTBDetection:
    """Result of OTB camera footage detection."""

    found: bool
    confidence: float = 0.0
    laplacian_variance: float = 0.0
    color_std: float = 0.0


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
    """Detect real OTB camera footage in the frame area outside the overlay.

    Analyzes the non-overlay portion of the frame for properties characteristic
    of real camera footage: high texture variance, color diversity, and
    irregular edge patterns.

    Args:
        frame: BGR video frame.
        overlay_bbox: (x, y, w, h) of the detected overlay region.

    Returns:
        OTBDetection with found=True if real camera footage is detected.
    """
    h, w = frame.shape[:2]
    ox, oy, ow, oh = overlay_bbox

    # Create a mask for the non-overlay region
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[oy : oy + oh, ox : ox + ow] = 0

    # Check that enough area remains outside the overlay
    otb_pixels = np.count_nonzero(mask)
    total_pixels = h * w
    if otb_pixels / total_pixels < MIN_OTB_AREA_FRACTION:
        return OTBDetection(found=False)

    # Extract the non-overlay region pixels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Laplacian variance — measures texture richness
    # Real scenes (wood grain, people, lighting) have high variance.
    # Solid backgrounds and rendered UIs have low variance.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_masked = laplacian[mask > 0]
    laplacian_var = float(np.var(laplacian_masked)) if len(laplacian_masked) > 0 else 0.0

    # 2. Color diversity — real scenes have more color variation
    otb_pixels_bgr = frame[mask > 0]
    color_std = float(np.std(otb_pixels_bgr)) if len(otb_pixels_bgr) > 0 else 0.0

    # Score based on both signals
    texture_score = min(laplacian_var / (MIN_LAPLACIAN_VARIANCE * 4), 1.0)
    color_score = min(color_std / (MIN_COLOR_STD * 4), 1.0)
    confidence = 0.6 * texture_score + 0.4 * color_score

    found = laplacian_var >= MIN_LAPLACIAN_VARIANCE and color_std >= MIN_COLOR_STD

    return OTBDetection(
        found=found,
        confidence=confidence,
        laplacian_variance=laplacian_var,
        color_std=color_std,
    )


def screen_video(video_url_or_path: str) -> ScreeningResult:
    """Screen a video for both overlay and OTB camera footage.

    Extracts 2-3 frames and checks each for:
    1. A 2D chess board overlay (via existing scanner)
    2. Real OTB camera footage outside the overlay

    Args:
        video_url_or_path: Local file path or YouTube URL.

    Returns:
        ScreeningResult with approved=True if both overlay and OTB detected.
    """
    import os
    frame_paths = extract_frames_from_video(video_url_or_path)

    if not frame_paths:
        logger.warning(f"Could not extract frames from {video_url_or_path}")
        return ScreeningResult(has_overlay=False, has_otb=False)

    best_overlay = OverlayDetection(found=False)
    best_otb = OTBDetection(found=False)
    best_overlay_bbox = None

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        # Check for overlay
        overlay_det = detect_overlay_in_frame(frame)
        if overlay_det.found and overlay_det.score > best_overlay.score:
            best_overlay = overlay_det

            # Check for OTB footage outside the overlay
            otb_det = detect_otb_region(frame, overlay_det.bbox)
            if otb_det.confidence > best_otb.confidence:
                best_otb = otb_det
                best_overlay_bbox = overlay_det.bbox

    # Clean up extracted frames
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
