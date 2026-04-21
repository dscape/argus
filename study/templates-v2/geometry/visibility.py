"""Stage-1 geometric visibility masks for templates-v2.

For every occupied square in a frame, project the piece's 3-D cuboid to image
space, compute its silhouette, subtract the silhouettes of all strictly closer
cuboids, and downsample the remaining visible region to a DINOv3 patch grid.
The resulting `patch_visibility` mask lets downstream stages weight patch
tokens by how much of each patch actually belongs to the piece.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import chess
import cv2
import numpy as np
from pipeline.physical.piece_projection import (
    CameraPose,
    board_to_image_homography,
    camera_pose_from_corners,
    default_camera_matrix,
    piece_bbox_from_projection,
)

DEFAULT_INPUT_SIZE = 224
DEFAULT_PATCH_SIZE = 16

# Per-piece cuboid dimensions in board units (1 unit = 1 square). Widths and
# depths are roughly proportional to the base diameter of a tournament Staunton
# set: pawns are narrow, the king/queen are the widest. Heights are
# proportional to real piece heights relative to a 55 mm square. Using
# per-type dimensions makes the cuboid cover the actual piece even when the
# piece is placed slightly off-center within its square.
PIECE_DIMENSIONS: dict[str, tuple[float, float, float]] = {
    "P": (0.45, 0.45, 1.20),
    "N": (0.55, 0.55, 1.50),
    "B": (0.50, 0.50, 1.75),
    "R": (0.60, 0.60, 1.40),
    "Q": (0.60, 0.60, 2.00),
    "K": (0.65, 0.65, 2.15),
}


@dataclass(frozen=True)
class PieceVisibility:
    square: str
    symbol: str
    row: int
    col: int
    depth: float
    cuboid_vertices: np.ndarray
    silhouette: np.ndarray
    crop_bbox: tuple[int, int, int, int]
    visibility_fraction: float
    patch_visibility: np.ndarray
    own_mask_full: np.ndarray | None = None
    visible_mask_full: np.ndarray | None = None

    @property
    def full_mask(self) -> np.ndarray | None:
        return self.own_mask_full

    @property
    def visible_mask(self) -> np.ndarray | None:
        return self.visible_mask_full

    @property
    def patch_visibility_mask(self) -> np.ndarray:
        return self.patch_visibility

    @property
    def crop_bbox_xyxy(self) -> tuple[int, int, int, int]:
        return self.crop_bbox


@dataclass(frozen=True)
class FrameVisibility:
    image_shape: tuple[int, int]
    pose: CameraPose
    fen: str
    input_size: int
    patch_size: int
    patch_grid_shape: tuple[int, int]
    occlusion_order: tuple[str, ...]
    pieces: Mapping[str, PieceVisibility]


def occupied_squares_from_fen(fen: str) -> list[tuple[str, str, int, int]]:
    board = chess.Board(fen if " " in fen else f"{fen} w - - 0 1")
    entries: list[tuple[str, str, int, int]] = []
    for square_index, piece in board.piece_map().items():
        file = chess.square_file(square_index)
        rank = chess.square_rank(square_index)
        row = 7 - rank
        col = file
        entries.append((chess.square_name(square_index), piece.symbol(), row, col))
    entries.sort(key=lambda entry: entry[0])
    return entries


RECTIFIED_BOARD_SIZE = 640
PIECE_SEARCH_ROWS_ABOVE = 1.2
PIECE_PERCENTILE = 15.0


def _empty_square_luminance(
    gray_rectified: np.ndarray,
    occupied: Sequence[tuple[str, str, int, int]],
    square_size_px: float,
) -> tuple[float, float]:
    """Estimate the board's dark/light luminance from squares that should be
    empty according to the FEN. Returns ``(dark_mean, light_mean)``."""
    occupied_mask = np.zeros((8, 8), dtype=bool)
    for _square, _symbol, row, col in occupied:
        occupied_mask[row, col] = True

    light_samples: list[float] = []
    dark_samples: list[float] = []
    for row in range(8):
        for col in range(8):
            if occupied_mask[row, col]:
                continue
            sx_min = int(round((col + 0.2) * square_size_px))
            sx_max = int(round((col + 0.8) * square_size_px))
            sy_min = int(round((row + 0.2) * square_size_px))
            sy_max = int(round((row + 0.8) * square_size_px))
            if sx_max <= sx_min or sy_max <= sy_min:
                continue
            patch = gray_rectified[sy_min:sy_max, sx_min:sx_max]
            if patch.size == 0:
                continue
            mean = float(patch.mean())
            if (row + col) % 2 == 0:
                light_samples.append(mean)
            else:
                dark_samples.append(mean)
    light_mean = float(np.median(light_samples)) if light_samples else 192.0
    dark_mean = float(np.median(dark_samples)) if dark_samples else 96.0
    return dark_mean, light_mean


def detect_piece_board_positions(
    image_bgr: np.ndarray,
    corners: Sequence[Sequence[float]],
    occupied: Sequence[tuple[str, str, int, int]],
    *,
    rectified_size: int = RECTIFIED_BOARD_SIZE,
    rows_above: float = PIECE_SEARCH_ROWS_ABOVE,
    percentile: float = PIECE_PERCENTILE,
) -> dict[str, tuple[float, float]]:
    """For each occupied square, detect the piece's board-space ``(cx, cy)``.

    Approach: rectify the board, learn the dark/light board luminance from
    empty squares, then threshold each occupied square's column strip with a
    piece-specific cutoff (extremes of the luminance histogram for white vs.
    black pieces). The piece base is taken as the median of the blob pixels
    inside the target square only — the blob's body above the square is used
    to confirm a detection but not to position it. Returns ``square -> (cx,
    cy)`` in board units. Falls back to geometric center on any failure.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")
    if rectified_size % 8 != 0:
        raise ValueError(f"rectified_size must be a multiple of 8, got {rectified_size}")

    corners_np = np.asarray(corners, dtype=np.float32)
    target = np.array(
        [
            [0.0, 0.0],
            [float(rectified_size), 0.0],
            [float(rectified_size), float(rectified_size)],
            [0.0, float(rectified_size)],
        ],
        dtype=np.float32,
    )
    homography_forward = cv2.getPerspectiveTransform(corners_np, target)
    rectified = cv2.warpPerspective(image_bgr, homography_forward, (rectified_size, rectified_size))
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    square_size_px = rectified_size / 8.0

    dark_mean, light_mean = _empty_square_luminance(gray, occupied, square_size_px)
    board_high = max(light_mean, dark_mean)
    board_low = min(light_mean, dark_mean)

    results: dict[str, tuple[float, float]] = {}
    for square_name, symbol, row, col in occupied:
        fallback = (col + 0.5, row + 0.5)
        sx_min = int(round(col * square_size_px))
        sx_max = int(round((col + 1) * square_size_px))
        sy_min = max(0, int(round((row - rows_above) * square_size_px)))
        sy_max = int(round((row + 1) * square_size_px))
        if sx_max <= sx_min or sy_max <= sy_min:
            results[square_name] = fallback
            continue

        region = gray[sy_min:sy_max, sx_min:sx_max]
        if region.size == 0:
            results[square_name] = fallback
            continue

        is_white_piece = symbol.isupper()
        if is_white_piece:
            cutoff_percentile = float(np.clip(100.0 - percentile, 50.0, 99.0))
            cutoff = float(np.percentile(region, cutoff_percentile))
            cutoff = max(cutoff, board_high + 10.0)
            mask = (region >= cutoff).astype(np.uint8)
        else:
            cutoff_percentile = float(np.clip(percentile, 1.0, 50.0))
            cutoff = float(np.percentile(region, cutoff_percentile))
            cutoff = min(cutoff, board_low - 10.0)
            mask = (region <= cutoff).astype(np.uint8)

        if mask.sum() < 12:
            results[square_name] = fallback
            continue

        target_top_local = int(round(row * square_size_px)) - sy_min
        target_top_local = max(0, target_top_local)
        target_mask = mask[target_top_local:]
        if target_mask.sum() < 6:
            # No piece-like pixels in the target square — keep geometric center.
            results[square_name] = fallback
            continue

        ys_target, xs_target = np.where(target_mask > 0)
        median_x_local = float(np.median(xs_target))
        median_y_local_in_target = float(np.median(ys_target))
        base_x_rect = sx_min + median_x_local
        base_y_rect = sy_min + target_top_local + median_y_local_in_target

        cx_board = float(base_x_rect) / square_size_px
        cy_board = float(base_y_rect) / square_size_px

        cx_board = float(np.clip(cx_board, col + 0.2, col + 0.8))
        cy_board = float(np.clip(cy_board, row + 0.2, row + 0.8))
        results[square_name] = (cx_board, cy_board)

    return results


def board_fen_from_piece_entries(entries: Sequence[Mapping[str, Any]]) -> str:
    board = chess.Board(None)
    for entry in entries:
        square_name = entry.get("square")
        symbol = entry.get("type")
        if not isinstance(square_name, str) or not isinstance(symbol, str):
            continue
        board.set_piece_at(chess.parse_square(square_name), chess.Piece.from_symbol(symbol))
    return board.board_fen()


def _silhouette_from_cuboid(cuboid_vertices: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(cuboid_vertices.astype(np.float32).reshape(-1, 1, 2))
    return hull.reshape(-1, 2).astype(np.float64)


def _compute_up_vanishing_point(pose: CameraPose) -> tuple[np.ndarray, bool]:
    """Return (vp_image, is_finite). ``vp_image`` is the image-space point
    where all 3D vertical lines converge. If the up-direction is parallel to
    the camera plane, the vanishing point is at infinity — in which case
    ``is_finite=False`` and ``vp_image`` is a direction vector instead."""
    up_camera = pose.K @ pose.R @ np.array(
        [0.0, 0.0, float(pose.up_sign)], dtype=np.float64
    )
    if abs(float(up_camera[2])) < 1e-8:
        direction = up_camera[:2]
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction = direction / norm
        return direction, False
    return up_camera[:2] / up_camera[2], True


def _project_base_via_homography(
    board_homography: np.ndarray,
    base_xy: np.ndarray,
) -> np.ndarray:
    """Map 4 board-plane (x, y) points to image pixels via the exact corner
    homography. Pixel-perfect on the annotated 4 corners."""
    homogeneous = np.hstack([base_xy, np.ones((base_xy.shape[0], 1))])
    projected = (board_homography @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def _project_piece_cuboid(
    pose: CameraPose,
    board_homography: np.ndarray,
    vp_image: np.ndarray,
    vp_is_finite: bool,
    *,
    cx: float,
    cy: float,
    width: float,
    depth: float,
    height: float,
) -> np.ndarray:
    """Project the 8 corners of a piece cuboid.

    Base corners (z=0) go through the exact board→image homography so they
    sit pixel-perfect on the playing surface. Top corners lie on the rays
    from each base corner toward the shared up vanishing point — that's
    where every 3D vertical line in the frame converges, regardless of
    where the camera is. Per-corner perspective scale is derived from the
    ``K[R|t]`` projection's homogeneous w-ratio so taller/closer pieces get
    longer image-space edges, shorter/further pieces get shorter edges.
    """
    half_w = width / 2.0
    half_d = depth / 2.0
    z_top = float(pose.up_sign) * float(height)
    base_xy = np.array(
        [
            [cx - half_w, cy - half_d],
            [cx + half_w, cy - half_d],
            [cx + half_w, cy + half_d],
            [cx - half_w, cy + half_d],
        ],
        dtype=np.float64,
    )
    base_image = _project_base_via_homography(board_homography, base_xy)

    if not vp_is_finite:
        # Parallel projection: top = base + fixed direction * height
        top_image = base_image + (vp_image * float(height))[None, :]
        return np.vstack([base_image, top_image])

    # For each base corner compute w_base and w_top under K[R|t], then
    # interpolate base→VP by k = 1 - w_base/w_top. Using the homography-
    # exact base keeps the cuboid on the square; interpolating toward VP
    # keeps all 4 edges converging consistently, even when the camera is
    # off-center.
    row2 = pose.R[2]
    base_3d = np.hstack([base_xy, np.zeros((4, 1))])
    top_3d = base_3d.copy()
    top_3d[:, 2] = z_top
    w_base = base_3d @ row2 + float(pose.t[2, 0])
    w_top = top_3d @ row2 + float(pose.t[2, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(np.abs(w_top) > 1e-8, 1.0 - w_base / w_top, 0.0)
    top_image = base_image + k[:, None] * (vp_image[None, :] - base_image)
    return np.vstack([base_image, top_image])


def _cuboid_center_depth(pose: CameraPose, cx: float, cy: float, height: float) -> float:
    center_board = np.array(
        [cx, cy, pose.up_sign * height / 2.0],
        dtype=np.float64,
    )
    camera_space = pose.R @ center_board + pose.t.reshape(-1)
    return float(camera_space[2])


def _clip_bbox_to_image(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x0 = int(np.clip(np.floor(bbox[0]), 0, width))
    y0 = int(np.clip(np.floor(bbox[1]), 0, height))
    x1 = int(np.clip(np.ceil(bbox[2]), 0, width))
    y1 = int(np.clip(np.ceil(bbox[3]), 0, height))
    return x0, y0, x1, y1


def _rasterize_silhouette(
    silhouette: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    if silhouette.size == 0:
        return mask
    polygon = np.round(silhouette).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def _place_on_canvas(
    pixels: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    *,
    input_size: int,
    flip_horizontally: bool,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Aspect-preserving crop + bottom-left paste into a square canvas.

    Shared transform used both for raster masks (single channel) and for
    image crops (three channels). The visibility mask and the crop image
    therefore live in the same pixel-to-patch coordinate system.
    """
    x0, y0, x1, y1 = crop_bbox
    if pixels.ndim == 3:
        canvas_shape: tuple[int, ...] = (input_size, input_size, pixels.shape[2])
    else:
        canvas_shape = (input_size, input_size)
    canvas = np.full(canvas_shape, pad_value, dtype=pixels.dtype)
    if x1 <= x0 or y1 <= y0:
        return canvas

    crop = pixels[y0:y1, x0:x1]
    crop_h = int(crop.shape[0])
    crop_w = int(crop.shape[1])
    scale = float(input_size) / float(max(crop_h, crop_w))
    new_h = max(1, int(round(crop_h * scale)))
    new_w = max(1, int(round(crop_w * scale)))
    interpolation = cv2.INTER_AREA if max(crop_h, crop_w) >= input_size else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    if flip_horizontally:
        resized = cv2.flip(resized, 1)
    y_offset = input_size - new_h
    canvas[y_offset : y_offset + new_h, 0:new_w] = resized
    return canvas


def _crop_mask_to_patch_grid(
    full_mask: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    *,
    input_size: int,
    patch_size: int,
    flip_horizontally: bool,
) -> np.ndarray:
    patch_count = input_size // patch_size
    canvas = _place_on_canvas(
        full_mask.astype(np.float32),
        crop_bbox,
        input_size=input_size,
        flip_horizontally=flip_horizontally,
        pad_value=0.0,
    )
    canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32)
    grid = canvas.reshape(patch_count, patch_size, patch_count, patch_size).mean(axis=(1, 3))
    return grid.astype(np.float32)


def extract_piece_crop(
    image_bgr: np.ndarray,
    piece: PieceVisibility,
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
    flip_left_half: bool = True,
) -> np.ndarray:
    """Extract the image crop corresponding to ``piece.crop_bbox``, placed on
    a ``(input_size, input_size, 3)`` canvas using the same aspect-preserving
    bottom-left paste as the visibility mask. The 14×14 geometric patch mask
    aligns with the 14×14 patch grid of this crop."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")
    flip = flip_left_half and piece.col < 4
    return _place_on_canvas(
        image_bgr,
        piece.crop_bbox,
        input_size=input_size,
        flip_horizontally=flip,
        pad_value=0.0,
    )


def _compute_frame_visibility_impl(
    *,
    image_shape: tuple[int, int] | tuple[int, int, int],
    corners: Sequence[Sequence[float]],
    fen: str,
    camera_matrix: np.ndarray | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    piece_positions: Mapping[str, tuple[float, float]] | None = None,
    piece_dimensions: Mapping[str, tuple[float, float, float]] | None = None,
    flip_left_half: bool = True,
    include_full_masks: bool = False,
) -> FrameVisibility:
    if len(image_shape) < 2:
        raise ValueError(f"image_shape must be (H, W) or (H, W, C), got {image_shape}")
    if input_size % patch_size != 0:
        raise ValueError(
            f"input_size ({input_size}) must be a multiple of patch_size ({patch_size})"
        )

    image_shape_2d = (int(image_shape[0]), int(image_shape[1]))
    resolved_camera_matrix = (
        default_camera_matrix(image_shape_2d) if camera_matrix is None else camera_matrix
    )
    pose = camera_pose_from_corners(corners, K=resolved_camera_matrix)
    board_homography = board_to_image_homography(corners)
    vp_image, vp_is_finite = _compute_up_vanishing_point(pose)
    positions = piece_positions or {}
    dimensions = {**PIECE_DIMENSIONS, **dict(piece_dimensions or {})}

    projections: list[dict[str, Any]] = []
    for square_name, symbol, row, col in occupied_squares_from_fen(fen):
        cx, cy = positions.get(square_name, (col + 0.5, row + 0.5))
        width, depth_size, height = dimensions[symbol.upper()]
        cuboid = _project_piece_cuboid(
            pose,
            board_homography,
            vp_image,
            vp_is_finite,
            cx=cx,
            cy=cy,
            width=width,
            depth=depth_size,
            height=height,
        )
        silhouette = _silhouette_from_cuboid(cuboid)
        depth = _cuboid_center_depth(pose, cx, cy, height)
        crop_bbox = _clip_bbox_to_image(piece_bbox_from_projection(cuboid), image_shape_2d)
        projections.append(
            {
                "square": square_name,
                "symbol": symbol,
                "row": row,
                "col": col,
                "center": (cx, cy),
                "depth": depth,
                "cuboid": cuboid,
                "silhouette": silhouette,
                "crop_bbox": crop_bbox,
            }
        )

    projections.sort(key=lambda entry: entry["depth"])

    own_masks: dict[str, np.ndarray] = {
        entry["square"]: _rasterize_silhouette(entry["silhouette"], image_shape_2d)
        for entry in projections
    }
    seen_mask = np.zeros(image_shape_2d, dtype=np.uint8)
    pieces: dict[str, PieceVisibility] = {}
    for entry in projections:
        square = entry["square"]
        own_mask = own_masks[square]
        visible_mask = np.where(seen_mask == 0, own_mask, np.uint8(0))
        np.bitwise_or(seen_mask, own_mask, out=seen_mask)

        own_area = int(own_mask.sum())
        visible_area = int(visible_mask.sum())
        visibility_fraction = (
            0.0 if own_area == 0 else float(visible_area) / float(own_area)
        )
        flip_horizontally = flip_left_half and entry["col"] < 4
        patch_visibility = _crop_mask_to_patch_grid(
            visible_mask,
            entry["crop_bbox"],
            input_size=input_size,
            patch_size=patch_size,
            flip_horizontally=flip_horizontally,
        )

        pieces[square] = PieceVisibility(
            square=square,
            symbol=entry["symbol"],
            row=entry["row"],
            col=entry["col"],
            depth=entry["depth"],
            cuboid_vertices=np.asarray(entry["cuboid"], dtype=np.float64),
            silhouette=entry["silhouette"],
            crop_bbox=entry["crop_bbox"],
            visibility_fraction=visibility_fraction,
            patch_visibility=patch_visibility,
            own_mask_full=own_mask.copy() if include_full_masks else None,
            visible_mask_full=visible_mask.copy() if include_full_masks else None,
        )

    patch_count = input_size // patch_size
    return FrameVisibility(
        image_shape=image_shape_2d,
        pose=pose,
        fen=fen,
        input_size=input_size,
        patch_size=patch_size,
        patch_grid_shape=(patch_count, patch_count),
        occlusion_order=tuple(entry["square"] for entry in projections),
        pieces=pieces,
    )


def compute_frame_visibility(
    corners: Sequence[Sequence[float]] | None = None,
    fen: str | None = None,
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    *,
    camera_matrix: np.ndarray | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    output_size: int | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    patch_grid_shape: tuple[int, int] | None = None,
    piece_positions: Mapping[str, tuple[float, float]] | None = None,
    piece_dimensions: Mapping[str, tuple[float, float, float]] | None = None,
    flip_left_half: bool = True,
    include_full_masks: bool = False,
    include_full_frame_masks: bool = False,
) -> FrameVisibility:
    resolved_input_size = output_size if output_size is not None else input_size
    if patch_grid_shape is not None:
        patch_rows, patch_cols = patch_grid_shape
        if patch_rows != patch_cols:
            raise ValueError(f"patch_grid_shape must be square, got {patch_grid_shape}")
        if patch_rows <= 0:
            raise ValueError(f"patch_grid_shape must be positive, got {patch_grid_shape}")
        if resolved_input_size % patch_rows != 0:
            raise ValueError(
                "output size "
                f"{resolved_input_size} is not divisible by patch grid {patch_grid_shape}"
            )
        patch_size = resolved_input_size // patch_rows

    if corners is None or fen is None or image_shape is None:
        raise TypeError("corners, fen, and image_shape are required")

    return _compute_frame_visibility_impl(
        image_shape=image_shape,
        corners=corners,
        fen=fen,
        camera_matrix=camera_matrix,
        input_size=resolved_input_size,
        patch_size=patch_size,
        piece_positions=piece_positions,
        piece_dimensions=piece_dimensions,
        flip_left_half=flip_left_half,
        include_full_masks=include_full_masks or include_full_frame_masks,
    )


def downsample_mask_to_patch_grid(
    crop_mask: np.ndarray,
    patch_grid_shape: tuple[int, int],
) -> np.ndarray:
    if crop_mask.ndim != 2:
        raise ValueError(f"crop_mask must be 2D, got {crop_mask.shape}")
    patch_rows, patch_cols = patch_grid_shape
    if patch_rows <= 0 or patch_cols <= 0:
        raise ValueError(f"patch_grid_shape must be positive, got {patch_grid_shape}")
    height, width = crop_mask.shape
    if height % patch_rows == 0 and width % patch_cols == 0:
        patch_h = height // patch_rows
        patch_w = width // patch_cols
        return crop_mask.reshape(patch_rows, patch_h, patch_cols, patch_w).mean(axis=(1, 3))
    resized = cv2.resize(
        crop_mask.astype(np.float32),
        (patch_cols, patch_rows),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float32)


def build_square_patch_visibility_mask(
    corners: Sequence[Sequence[float]],
    fen: str,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    square: str,
    *,
    camera_matrix: np.ndarray | None = None,
    output_size: int = DEFAULT_INPUT_SIZE,
    patch_grid_shape: tuple[int, int] | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    piece_positions: Mapping[str, tuple[float, float]] | None = None,
    piece_dimensions: Mapping[str, tuple[float, float, float]] | None = None,
    flip_left_half: bool = True,
) -> np.ndarray:
    frame_visibility = compute_frame_visibility(
        corners,
        fen,
        frame_shape,
        camera_matrix=camera_matrix,
        output_size=output_size,
        patch_grid_shape=patch_grid_shape,
        patch_size=patch_size,
        piece_positions=piece_positions,
        piece_dimensions=piece_dimensions,
        flip_left_half=flip_left_half,
    )
    return np.asarray(frame_visibility.pieces[square].patch_visibility, dtype=np.float32)


def without_full_masks(frame: FrameVisibility) -> FrameVisibility:
    """Return a copy with per-piece full-frame masks dropped (for serialization)."""
    stripped = {
        square: replace(piece, own_mask_full=None, visible_mask_full=None)
        for square, piece in frame.pieces.items()
    }
    return replace(frame, pieces=stripped)


__all__ = [
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_PATCH_SIZE",
    "PIECE_DIMENSIONS",
    "RECTIFIED_BOARD_SIZE",
    "FrameVisibility",
    "PieceVisibility",
    "board_fen_from_piece_entries",
    "build_square_patch_visibility_mask",
    "compute_frame_visibility",
    "detect_piece_board_positions",
    "downsample_mask_to_patch_grid",
    "extract_piece_crop",
    "occupied_squares_from_fen",
    "without_full_masks",
]
