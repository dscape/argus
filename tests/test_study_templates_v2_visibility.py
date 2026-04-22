from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _PROJECT_ROOT / "study" / "templates-v2" / "geometry" / "visibility.py"
_MODULE_NAME = "study_templates_v2_visibility"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_MODULE_PATH}")
visibility = importlib.util.module_from_spec(_SPEC)
__import__("sys").modules[_MODULE_NAME] = visibility
_SPEC.loader.exec_module(visibility)

_FRAME_SHAPE = (1080, 1920, 3)


def _project_with_cv2(rvec: np.ndarray, tvec: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    K = visibility.default_camera_matrix(_FRAME_SHAPE)
    projected, _ = cv2.projectPoints(
        points_3d.astype(np.float32),
        rvec.astype(np.float32),
        tvec.astype(np.float32),
        K.astype(np.float32),
        np.zeros(4, dtype=np.float32),
    )
    return projected.reshape(-1, 2).astype(np.float64)


def _camera_rvec_tvec(
    camera_pos_world: np.ndarray, look_at_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    forward = look_at_world - camera_pos_world
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    cam_down = np.cross(forward, right)
    cam_down = cam_down / np.linalg.norm(cam_down)
    rotation_world_to_camera = np.stack([right, cam_down, forward], axis=0)
    rvec, _ = cv2.Rodrigues(rotation_world_to_camera)
    tvec = -rotation_world_to_camera @ camera_pos_world
    return rvec.flatten().astype(np.float32), tvec.astype(np.float32)


def _synthetic_corners(camera_pos_world: np.ndarray) -> tuple[tuple[float, float], ...]:
    rvec, tvec = _camera_rvec_tvec(camera_pos_world, np.array([4.0, 4.0, 0.0], dtype=np.float64))
    board_corners = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [8.0, 8.0, 0.0], [0.0, 8.0, 0.0]])
    points = _project_with_cv2(rvec, tvec, board_corners)
    return tuple((float(point[0]), float(point[1])) for point in points)


@pytest.mark.skip(reason="Temporarily skipped while templates-v2 visibility geometry is in flux")
def test_front_pawn_stays_fully_visible_while_back_rank_rook_is_occluded() -> None:
    corners = _synthetic_corners(np.array([4.0, -7.0, 4.0], dtype=np.float64))
    fen = "4r3/4p3/8/8/8/8/8/8"

    frame_visibility = visibility.compute_frame_visibility(
        corners,
        fen,
        _FRAME_SHAPE,
        patch_grid_shape=(16, 16),
        include_full_frame_masks=True,
    )

    front_pawn = frame_visibility.pieces["e7"]
    back_rook = frame_visibility.pieces["e8"]

    assert front_pawn.visibility_fraction > 0.99
    assert 0.10 < back_rook.visibility_fraction < 0.95
    assert front_pawn.depth < back_rook.depth
    assert np.any(back_rook.full_mask.astype(bool) & ~back_rook.visible_mask.astype(bool))


def test_downsample_mask_to_patch_grid_averages_patch_coverage() -> None:
    crop_mask = np.zeros((8, 8), dtype=np.float32)
    crop_mask[:, :4] = 1.0

    patch_mask = visibility.downsample_mask_to_patch_grid(crop_mask, (2, 2))

    np.testing.assert_allclose(patch_mask, np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))


def test_square_patch_visibility_mask_returns_fractional_grid_values() -> None:
    corners = _synthetic_corners(np.array([4.0, -7.0, 4.0], dtype=np.float64))
    fen = "4r3/4p3/8/8/8/8/8/8"

    patch_mask = visibility.build_square_patch_visibility_mask(
        corners,
        fen,
        _FRAME_SHAPE,
        "e8",
        patch_grid_shape=(16, 16),
    )

    assert patch_mask.shape == (16, 16)
    assert float(patch_mask.min()) >= 0.0
    assert float(patch_mask.max()) <= 1.0
    assert float(patch_mask.sum()) > 0.0
