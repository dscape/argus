"""Assign piece detections to squares on the rectified board.

Detectors output 2D bounding boxes in the tight (no-margin) rectified 512x512
board space. Because pieces tip upward in the image but their **base** sits on
the actual chess square, we use the **lower half** of each box to decide
placement: the IoU between the bottom-half of a detection and each of the 64
square tiles. The square with the highest IoU (above ``iou_threshold``) wins.

For collisions (two detections claiming the same square), we keep the one with
the higher score.

Boxes are in pixel coordinates ``(x_min, y_min, x_max, y_max)`` on a
``board_size`` x ``board_size`` canvas laid out in the usual row-major order
(row 0 = rank 8, row 7 = rank 1; col 0 = file a, col 7 = file h).
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.physical.square_classifiers import piece_label_to_square_class


@dataclass(frozen=True)
class PieceDetection:
    """One detector output: 12-class piece label, pixel bbox, confidence score."""

    piece_label: int  # 0..11, maps to SQUARE_CLASS_NAMES[piece_label + 1]
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float


def assign_detections_to_squares(
    detections: list[PieceDetection],
    *,
    board_size: int = 512,
    iou_threshold: float = 0.3,
) -> tuple[int, ...]:
    """Return 64 class ids in SQUARE_CLASS_NAMES space (0 = empty).

    Each detection contributes to the square whose tile has the highest
    bottom-half IoU with the detection's bbox, provided the IoU meets
    ``iou_threshold``. When two detections select the same square, the one
    with the higher ``score`` wins.
    """
    square_size = board_size / 8.0
    # Track the best score per square so later collisions overwrite only if better.
    best_by_square: dict[int, tuple[float, int]] = {}

    for detection in detections:
        lower_half = _lower_half(detection)
        if lower_half is None:
            continue
        best_square = -1
        best_iou = 0.0
        for square_index in range(64):
            square_box = _square_box(square_index, square_size=square_size)
            iou = _iou(lower_half, square_box)
            if iou > best_iou:
                best_iou = iou
                best_square = square_index
        if best_square < 0 or best_iou < iou_threshold:
            continue
        existing = best_by_square.get(best_square)
        if existing is None or detection.score > existing[0]:
            best_by_square[best_square] = (
                detection.score,
                piece_label_to_square_class(detection.piece_label),
            )

    class_ids = [0] * 64
    for square_index, (_score, class_id) in best_by_square.items():
        class_ids[square_index] = class_id
    return tuple(class_ids)


def _lower_half(detection: PieceDetection) -> tuple[float, float, float, float] | None:
    width = detection.xmax - detection.xmin
    height = detection.ymax - detection.ymin
    if width <= 0.0 or height <= 0.0:
        return None
    midpoint = detection.ymin + height / 2.0
    return (detection.xmin, midpoint, detection.xmax, detection.ymax)


def _square_box(square_index: int, *, square_size: float) -> tuple[float, float, float, float]:
    row = square_index // 8
    col = square_index % 8
    xmin = col * square_size
    ymin = row * square_size
    return (xmin, ymin, xmin + square_size, ymin + square_size)


def _iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
