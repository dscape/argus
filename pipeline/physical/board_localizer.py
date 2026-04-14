"""Physical-board localization for real broadcast frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.screen.otb_yolo_detector import detect_otb_yolo


@dataclass(frozen=True)
class BoardLocalization:
    """Physical-board location in one frame."""

    corners: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    confidence: float
    method: str


def localize_board(frame: np.ndarray, *, device: str = "cpu") -> BoardLocalization | None:
    """Return best-effort board corners for one physical broadcast frame."""
    yolo_detection = detect_otb_yolo(frame, device=device)
    if yolo_detection.found and yolo_detection.bbox is not None:
        return BoardLocalization(
            corners=_bbox_to_corners(yolo_detection.bbox),
            confidence=float(yolo_detection.confidence),
            method="otb_yolo",
        )

    alternation_bbox, alternation_score = _localize_with_alternation_search(frame)
    if alternation_bbox is not None:
        return BoardLocalization(
            corners=_bbox_to_corners(alternation_bbox),
            confidence=float(alternation_score),
            method="alternation_search",
        )

    contour_bbox = _localize_with_contours(frame)
    if contour_bbox is not None:
        return BoardLocalization(
            corners=_bbox_to_corners(contour_bbox),
            confidence=0.0,
            method="contour",
        )
    return None


def _localize_with_alternation_search(
    frame: np.ndarray,
) -> tuple[tuple[int, int, int, int] | None, float]:
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return None, 0.0

    target_max_dim = 360
    scale = min(float(target_max_dim) / max(height, width), 1.0)
    if scale < 1.0:
        resized = cv2.resize(
            frame,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = frame
    search_h, search_w = resized.shape[:2]

    best_bbox: tuple[int, int, int, int] | None = None
    best_score = float("-inf")
    for window_fraction in (0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.76):
        window = int(min(search_h, search_w) * window_fraction)
        if window < 48:
            continue
        step = max(window // 8, 6)
        for y in range(0, search_h - window + 1, step):
            for x in range(0, search_w - window + 1, step):
                score = _checkerboard_window_score(resized[y : y + window, x : x + window])
                if score > best_score:
                    best_score = score
                    best_bbox = (x, y, window, window)

    if best_bbox is None or best_score <= 0.0:
        return None, 0.0

    expanded = _expand_bbox(best_bbox, width=search_w, height=search_h, scale=1.35)
    if scale < 1.0:
        x, y, w, h = expanded
        expanded = (
            int(round(x / scale)),
            int(round(y / scale)),
            int(round(w / scale)),
            int(round(h / scale)),
        )
    return _clip_bbox(expanded, width=width, height=height), float(best_score)


def _localize_with_contours(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = frame.shape[:2]
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
            aspect = w / h if h > 0 else 0.0
            if 0.6 <= aspect <= 1.6:
                best_area = area
                best_rect = (x, y, w, h)

    return best_rect


def _checkerboard_window_score(region_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    cell = min(height, width) // 8
    if cell < 4:
        return float("-inf")

    trimmed = gray[: cell * 8, : cell * 8]
    means = trimmed.reshape(8, cell, 8, cell).transpose(0, 2, 1, 3).reshape(8, 8, -1).mean(axis=2)
    horizontal = np.abs(means[:, :-1] - means[:, 1:])
    vertical = np.abs(means[:-1, :] - means[1:, :])
    diffs = np.concatenate([horizontal.reshape(-1), vertical.reshape(-1)])
    alternation_fraction = float((diffs > 15.0).mean())
    contrast = float(diffs.mean())

    same_color_mask = (np.indices((8, 8)).sum(axis=0) % 2) == 0
    same_color_std = float(means[same_color_mask].std() + means[~same_color_mask].std())
    return alternation_fraction * contrast - 0.5 * same_color_std


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    expanded_w = w * scale
    expanded_h = h * scale
    expanded = (
        int(round(center_x - expanded_w / 2.0)),
        int(round(center_y - expanded_h / 2.0)),
        int(round(expanded_w)),
        int(round(expanded_h)),
    )
    return _clip_bbox(expanded, width=width, height=height)


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _bbox_to_corners(
    bbox: tuple[int, int, int, int],
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    x, y, w, h = bbox
    return (
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x + w), float(y + h)),
        (float(x), float(y + h)),
    )
