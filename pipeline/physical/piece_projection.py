"""Project 3D piece bounding boxes to image space using the 4 board corners.

The idea:

1. The 4 image-space corners of the playing surface give us a board → image
   homography ``H``.
2. With an assumed camera intrinsics matrix ``K``, ``cv2.decomposeHomographyMat``
   recovers the full camera pose ``[R|t]`` up to a 4-way ambiguity; we keep the
   pose whose plane normal points back toward the camera.
3. A piece on square ``(row, col)`` occupies the 3D box
   ``[col, col+1] × [row, row+1] × [0, piece_height]`` in board units (where
   ``1 unit = 1 square``).
4. Project that 3D box through ``P = K [R|t]``. The 8 image-space corners define
   the piece's image bounding quad. An axis-aligned bbox gives a quick crop;
   the full quad lets us do a perspective warp to a fixed output canvas.

If the camera matches chesscog's (~45° oblique, moderate distance), the
projected box matches chesscog's top/side extensions. For a near-top-down
camera the projected box collapses toward the square itself. The logic adapts
automatically — no per-setup tuning required.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Chess pieces are ~1.5 board units tall (king ~10 cm, square ~6 cm).
# Use 2.0 as a conservative upper bound so the crop always contains the full piece.
DEFAULT_PIECE_HEIGHT = 2.0

_BOARD_UNITS = 8.0


@dataclass(frozen=True)
class CameraPose:
    """Recovered camera pose for one frame."""

    K: np.ndarray  # (3, 3) intrinsics
    R: np.ndarray  # (3, 3) rotation, board frame -> camera frame
    t: np.ndarray  # (3, 1) translation, board frame origin -> camera frame
    normal: np.ndarray  # (3,) board-plane normal in camera frame

    @property
    def projection(self) -> np.ndarray:
        """``P = K [R|t]`` as a 3x4 matrix."""
        return self.K @ np.hstack([self.R, self.t])


@dataclass(frozen=True)
class BoardNeighborhoodCrop:
    """Axis-aligned crop around the annotated board and its relative corners."""

    image_bgr: np.ndarray
    corners: np.ndarray
    x1: int
    y1: int


def default_camera_matrix(frame_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    """Default pinhole intrinsics for a frame of the given shape.

    Assumes a 60° horizontal field of view, square pixels, and principal point
    at the image center. Good enough for homography decomposition when real
    intrinsics aren't available.
    """
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    focal = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def board_to_image_homography(
    corners: tuple[tuple[float, float], ...] | list[list[float]],
) -> np.ndarray:
    """Homography from board coords (0..8 units) to image coords."""
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {points.shape}")
    board_points = np.array(
        [
            [0.0, 0.0],
            [_BOARD_UNITS, 0.0],
            [_BOARD_UNITS, _BOARD_UNITS],
            [0.0, _BOARD_UNITS],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(board_points, points).astype(np.float64)


def extract_board_neighborhood_crop(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    crop_margin: float = 0.18,
) -> BoardNeighborhoodCrop:
    """Crop the frame down to the board neighborhood and return relative corners."""
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {points.shape}")
    if crop_margin < 0.0:
        raise ValueError(f"crop_margin must be >= 0, got {crop_margin}")

    height, width = image_bgr.shape[:2]
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    extent = np.maximum(max_xy - min_xy, 1.0)
    margin = extent * float(crop_margin)

    x1 = max(0, int(np.floor(min_xy[0] - margin[0])))
    y1 = max(0, int(np.floor(min_xy[1] - margin[1])))
    x2 = min(width, int(np.ceil(max_xy[0] + margin[0])))
    y2 = min(height, int(np.ceil(max_xy[1] + margin[1])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid board crop derived from corners")

    cropped = image_bgr[y1:y2, x1:x2].copy()
    relative_corners = points - np.array([x1, y1], dtype=np.float32)
    return BoardNeighborhoodCrop(image_bgr=cropped, corners=relative_corners, x1=x1, y1=y1)


def project_square_base_quad(
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    row: int,
    col: int,
) -> np.ndarray:
    """Project one square's base quad (z=0) into image pixels."""
    _validate_row_col(row, col)
    board_quad = np.array(
        [
            [float(col), float(row)],
            [float(col + 1), float(row)],
            [float(col + 1), float(row + 1)],
            [float(col), float(row + 1)],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    projected = cv2.perspectiveTransform(board_quad, board_to_image_homography(corners))
    return projected.reshape(4, 2).astype(np.float64)


def square_bbox_from_corners(
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    row: int,
    col: int,
    pad_ratio: float = 0.0,
) -> tuple[float, float, float, float]:
    """Axis-aligned image bbox for one square's projected base quad."""
    if pad_ratio < 0.0:
        raise ValueError(f"pad_ratio must be >= 0, got {pad_ratio}")
    projected = project_square_base_quad(corners, row=row, col=col)
    xmin = float(projected[:, 0].min())
    ymin = float(projected[:, 1].min())
    xmax = float(projected[:, 0].max())
    ymax = float(projected[:, 1].max())
    pad_x = (xmax - xmin) * pad_ratio
    pad_y = (ymax - ymin) * pad_ratio
    return xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y


def camera_pose_from_corners(
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    K: np.ndarray,
) -> CameraPose:
    """Recover the camera pose from the 4 known 3D->2D board corner correspondences.

    IPPE on 4 coplanar points has a sign ambiguity -- two R/t solutions reproject
    the same corners. We run ``cv2.solvePnPGeneric`` to get both candidates and
    keep the one whose board-plane normal has the SMALLER z in camera coords.
    That solution corresponds to the camera viewing the board from the +z side
    of the board plane in our world frame (so ``piece_height > 0`` moves
    upward, toward the camera, in image space).

    Note: even after picking the "better" solution, the sign of piece lift in
    world coords can still be positive or negative depending on the corner
    winding order. ``project_piece_box`` handles that at projection time.
    """
    image_points = np.asarray(corners, dtype=np.float32)
    if image_points.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {image_points.shape}")
    board_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [_BOARD_UNITS, 0.0, 0.0],
            [_BOARD_UNITS, _BOARD_UNITS, 0.0],
            [0.0, _BOARD_UNITS, 0.0],
        ],
        dtype=np.float32,
    )
    distortion = np.zeros(4, dtype=np.float32)
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        board_points,
        image_points,
        K.astype(np.float32),
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success or not rvecs:
        raise ValueError(f"solvePnPGeneric failed for corners {corners}")
    best_pose: CameraPose | None = None
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        R64 = R.astype(np.float64)
        t64 = tvec.astype(np.float64).reshape(3, 1)
        normal = R64 @ np.array([0.0, 0.0, 1.0])
        if float(t64[2, 0]) <= 0:
            continue
        if best_pose is None or normal[2] < best_pose.normal[2]:
            best_pose = CameraPose(K=K.astype(np.float64), R=R64, t=t64, normal=normal)
    if best_pose is None:
        raise ValueError(f"solvePnPGeneric gave no front-facing solution for {corners}")
    return best_pose


def project_points(pose: CameraPose, points_3d: np.ndarray) -> np.ndarray:
    """Project Nx3 board-frame points (board units, z=height) to Nx2 image pixels."""
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points_3d.shape}")
    homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    projected = (pose.projection @ homogeneous.T).T
    # Guard against the tiny w values that sometimes show up for points near the camera plane.
    w = projected[:, 2:3]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    return projected[:, :2] / w


def project_points_with_base_homography(
    pose: CameraPose,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    points_3d: np.ndarray,
) -> np.ndarray:
    """Like ``project_points`` but uses the exact 4-point homography for z=0 points.

    The 4-point homography has zero residual on the board corners -- it maps
    them exactly to their annotated image positions. For points at z=0 (the
    board surface) we use this homography, which eliminates the solvePnP
    intrinsics residual (~10-20 px for default K). Points at z!=0 still use
    the full camera [R|t] projection.
    """
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points_3d.shape}")
    H = cv2.getPerspectiveTransform(
        np.array(
            [[0.0, 0.0], [_BOARD_UNITS, 0.0], [_BOARD_UNITS, _BOARD_UNITS], [0.0, _BOARD_UNITS]],
            dtype=np.float32,
        ),
        np.asarray(corners, dtype=np.float32),
    ).astype(np.float64)
    on_plane_mask = np.abs(points_3d[:, 2]) < 1e-9
    result = np.empty((points_3d.shape[0], 2), dtype=np.float64)
    if on_plane_mask.any():
        planar = points_3d[on_plane_mask, :2]
        planar_h = np.hstack([planar, np.ones((planar.shape[0], 1))])
        transformed = (H @ planar_h.T).T
        result[on_plane_mask] = transformed[:, :2] / transformed[:, 2:3]
    off_plane_mask = ~on_plane_mask
    if off_plane_mask.any():
        result[off_plane_mask] = project_points(pose, points_3d[off_plane_mask])
    return result


def project_piece_box(
    pose: CameraPose,
    *,
    row: int,
    col: int,
    piece_height: float = DEFAULT_PIECE_HEIGHT,
    corners: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
) -> np.ndarray:
    """Project the 8 corners of the piece's 3D bounding box to image space.

    Returns an array of shape (8, 2): indices 0..3 are the base (z=0) in
    board-coord order (TL, TR, BR, BL) and 4..7 are the corresponding top
    (``piece_height`` above the board) corners. "Above" is determined from the
    recovered pose -- see below.

    Sign convention: the annotation's corner winding order fixes which way
    world +z points. For some annotations +z is "up" (away from board toward
    camera), for others +z is "down" (into the ground). We detect this from
    the pose: if the board-plane normal in camera coords points away from the
    camera (``normal[2] > 0``), then world +z is into the ground, so we lift
    the piece in -z instead. Either way, the image projection ends up with
    the piece top on the camera-facing side of the board.
    """
    up_sign = -1.0 if float(pose.normal[2]) > 0 else 1.0
    base = np.array(
        [
            [col, row, 0.0],
            [col + 1, row, 0.0],
            [col + 1, row + 1, 0.0],
            [col, row + 1, 0.0],
        ],
        dtype=np.float64,
    )
    top = base.copy()
    top[:, 2] = up_sign * float(piece_height)
    points = np.vstack([base, top])
    if corners is not None:
        return project_points_with_base_homography(pose, corners, points)
    return project_points(pose, points)


def piece_bbox_from_projection(projected_quad: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned (xmin, ymin, xmax, ymax) from projected piece-box corners."""
    xs = projected_quad[:, 0]
    ys = projected_quad[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def transform_projected_bboxes_to_crop_space(
    projected_bboxes: np.ndarray,
    crop: BoardNeighborhoodCrop,
    *,
    output_shape: tuple[int, int] | int,
) -> np.ndarray:
    """Map full-frame projected piece boxes into a cropped/resized board view."""
    if projected_bboxes.ndim != 2 or projected_bboxes.shape[1] != 4:
        raise ValueError(
            f"projected_bboxes must have shape (N, 4), got {projected_bboxes.shape}"
        )
    if isinstance(output_shape, int):
        output_height = output_width = int(output_shape)
    else:
        output_height = int(output_shape[0])
        output_width = int(output_shape[1])

    crop_height, crop_width = crop.image_bgr.shape[:2]
    scale_x = float(output_width) / max(float(crop_width), 1.0)
    scale_y = float(output_height) / max(float(crop_height), 1.0)

    transformed = projected_bboxes.astype(np.float64, copy=True)
    transformed[:, [0, 2]] -= float(crop.x1)
    transformed[:, [1, 3]] -= float(crop.y1)
    transformed[:, [0, 2]] *= scale_x
    transformed[:, [1, 3]] *= scale_y
    return transformed


def project_piece_bboxes(
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    piece_height: float = DEFAULT_PIECE_HEIGHT,
    K: np.ndarray | None = None,
) -> np.ndarray:
    """Project all 64 piece boxes and return their axis-aligned image bboxes.

    Output shape is ``(64, 4)`` in row-major square order with columns
    ``(xmin, ymin, xmax, ymax)``.
    """
    if K is None:
        K = default_camera_matrix(frame_shape)
    pose = camera_pose_from_corners(corners, K=K)
    bboxes = np.empty((64, 4), dtype=np.float64)
    for square_index in range(64):
        projected = project_piece_box(
            pose,
            row=square_index // 8,
            col=square_index % 8,
            piece_height=piece_height,
            corners=corners,
        )
        bboxes[square_index] = piece_bbox_from_projection(projected)
    return bboxes


def _validate_row_col(row: int, col: int) -> None:
    if not 0 <= row <= 7 or not 0 <= col <= 7:
        raise ValueError(f"row and col must be in [0, 7], got row={row}, col={col}")


def _axis_aligned_crop(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    output_size: int,
    flip_horizontally: bool,
) -> np.ndarray:
    """Resize a clipped-to-frame axis-aligned bbox to a fixed square canvas.

    Preserves the aspect of the bbox — wider bboxes keep width filling the
    canvas, taller bboxes keep height filling the canvas — and pastes the
    result at the bottom-left so the square baseline lines up consistently
    across samples.
    """
    xmin, ymin, xmax, ymax = bbox
    height, width = image_bgr.shape[:2]
    clipped_xmin = max(0, int(np.floor(xmin)))
    clipped_ymin = max(0, int(np.floor(ymin)))
    clipped_xmax = min(width, int(np.ceil(xmax)))
    clipped_ymax = min(height, int(np.ceil(ymax)))
    if clipped_xmax <= clipped_xmin or clipped_ymax <= clipped_ymin:
        return np.zeros((output_size, output_size, image_bgr.shape[2]), dtype=image_bgr.dtype)

    crop = image_bgr[clipped_ymin:clipped_ymax, clipped_xmin:clipped_xmax]
    crop_h, crop_w = crop.shape[:2]
    scale = float(output_size) / float(max(crop_h, crop_w))
    new_h = max(1, int(round(crop_h * scale)))
    new_w = max(1, int(round(crop_w * scale)))
    interpolation = cv2.INTER_AREA if max(crop_h, crop_w) >= output_size else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    if flip_horizontally:
        resized = cv2.flip(resized, 1)

    canvas = np.zeros((output_size, output_size, image_bgr.shape[2]), dtype=image_bgr.dtype)
    y_offset = output_size - new_h
    canvas[y_offset : y_offset + new_h, 0:new_w] = resized
    return canvas


def extract_projected_piece_crop(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    row: int,
    col: int,
    output_size: int,
    piece_height: float = DEFAULT_PIECE_HEIGHT,
    K: np.ndarray | None = None,
    flip_left_half: bool = True,
) -> np.ndarray:
    """Extract a piece-classifier crop using the projected 3D bounding box.

    Unlike chesscog's fixed per-rank/file extensions, the crop size adapts to
    the actual camera angle recovered from the 4 board corners.
    """
    if K is None:
        K = default_camera_matrix(image_bgr.shape)
    pose = camera_pose_from_corners(corners, K=K)
    projected = project_piece_box(pose, row=row, col=col, piece_height=piece_height, corners=corners)
    bbox = piece_bbox_from_projection(projected)
    flip = flip_left_half and col < 4
    return _axis_aligned_crop(image_bgr, bbox, output_size=output_size, flip_horizontally=flip)


def extract_projected_occupancy_crop(
    image_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    row: int,
    col: int,
    output_size: int,
    K: np.ndarray | None = None,
    pad_ratio: float = 0.3,
) -> np.ndarray:
    """Occupancy crop: just the square's base quad plus a small symmetric pad."""
    del K
    bbox = square_bbox_from_corners(
        corners,
        row=row,
        col=col,
        pad_ratio=pad_ratio,
    )
    return _axis_aligned_crop(image_bgr, bbox, output_size=output_size, flip_horizontally=False)
