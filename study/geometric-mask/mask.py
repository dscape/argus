from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pipeline.physical.piece_projection import (
    DEFAULT_PIECE_HEIGHT,
    camera_pose_from_corners,
    default_camera_matrix,
    piece_bbox_from_projection,
    project_piece_box,
)


@dataclass(frozen=True)
class GeometricMaskBuilder:
    corners: tuple[tuple[float, float], ...]
    frame_shape: tuple[int, ...]
    output_size: int
    piece_height: float = DEFAULT_PIECE_HEIGHT

    def __post_init__(self) -> None:
        K = default_camera_matrix(self.frame_shape)
        pose = camera_pose_from_corners(self.corners, K=K)
        projected_boxes = []
        bbox_array = np.zeros((64, 4), dtype=np.float64)
        depths = np.zeros((64,), dtype=np.float64)
        for square_index in range(64):
            row = square_index // 8
            col = square_index % 8
            projected = project_piece_box(
                pose,
                row=row,
                col=col,
                piece_height=self.piece_height,
                corners=self.corners,
            )
            projected_boxes.append(projected)
            bbox_array[square_index] = piece_bbox_from_projection(projected)
            center = np.array([col + 0.5, row + 0.5, self.piece_height * 0.5], dtype=np.float64)
            camera_point = pose.R @ center + pose.t.reshape(3)
            depths[square_index] = float(camera_point[2])
        object.__setattr__(self, "_projected_boxes", tuple(projected_boxes))
        object.__setattr__(self, "_bboxes", bbox_array)
        object.__setattr__(self, "_depths", depths)
        object.__setattr__(self, "_mask_cache", {})

    def mask_for_square(self, *, square_index: int) -> np.ndarray:
        cached = self._mask_cache.get(square_index)
        if cached is not None:
            return cached.copy()
        target_depth = float(self._depths[square_index])
        closer_polygons = []
        for other_square in range(64):
            if other_square == square_index or float(self._depths[other_square]) >= target_depth:
                continue
            hull = cv2.convexHull(self._projected_boxes[other_square].astype(np.float32))
            closer_polygons.append(hull.reshape(-1, 2))
        flip_horizontally = square_index % 8 < 4
        mask = build_mask_from_closer_polygons(
            target_bbox=self._bboxes[square_index],
            closer_polygons=closer_polygons,
            output_size=self.output_size,
            flip_horizontally=flip_horizontally,
        )
        self._mask_cache[square_index] = mask.copy()
        return mask

    def apply_to_crop(self, crop_bgr: np.ndarray, *, square_index: int) -> np.ndarray:
        mask = self.mask_for_square(square_index=square_index)
        if crop_bgr.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(
                mask,
                (crop_bgr.shape[1], crop_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        result = crop_bgr.copy()
        result[mask <= 0] = 0
        return result


def build_mask_from_closer_polygons(
    *,
    target_bbox: np.ndarray,
    closer_polygons: list[np.ndarray],
    output_size: int,
    flip_horizontally: bool,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in target_bbox.tolist()]
    clipped_x1 = int(np.floor(x1))
    clipped_y1 = int(np.floor(y1))
    clipped_x2 = int(np.ceil(x2))
    clipped_y2 = int(np.ceil(y2))
    crop_width = max(1, clipped_x2 - clipped_x1)
    crop_height = max(1, clipped_y2 - clipped_y1)
    local_mask = np.full((crop_height, crop_width), 255, dtype=np.uint8)
    for polygon in closer_polygons:
        local_polygon = np.asarray(polygon, dtype=np.float32).copy()
        local_polygon[:, 0] -= float(clipped_x1)
        local_polygon[:, 1] -= float(clipped_y1)
        hull = cv2.convexHull(local_polygon).astype(np.int32)
        cv2.fillConvexPoly(local_mask, hull, 0)
    return project_local_mask_to_crop_canvas(
        local_mask,
        output_size=output_size,
        flip_horizontally=flip_horizontally,
    )


def project_local_mask_to_crop_canvas(
    local_mask: np.ndarray,
    *,
    output_size: int,
    flip_horizontally: bool,
) -> np.ndarray:
    crop_height, crop_width = local_mask.shape[:2]
    scale = float(output_size) / float(max(crop_height, crop_width))
    new_height = max(1, int(round(crop_height * scale)))
    new_width = max(1, int(round(crop_width * scale)))
    resized = cv2.resize(local_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    if flip_horizontally:
        resized = cv2.flip(resized, 1)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    y_offset = output_size - new_height
    canvas[y_offset : y_offset + new_height, 0:new_width] = resized
    return canvas


__all__ = [
    "GeometricMaskBuilder",
    "build_mask_from_closer_polygons",
    "project_local_mask_to_crop_canvas",
]
