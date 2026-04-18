from __future__ import annotations

import cv2
import numpy as np
from pipeline.physical.piece_projection import (
    DEFAULT_PIECE_HEIGHT,
    board_to_image_homography,
    camera_pose_from_corners,
    default_camera_matrix,
    extract_board_neighborhood_crop,
    piece_bbox_from_projection,
    project_piece_bboxes,
    project_piece_box,
    project_points,
    project_square_base_quad,
    square_bbox_from_corners,
    transform_projected_bboxes_to_crop_space,
)

_FRAME_SHAPE = (1080, 1920, 3)
_K = default_camera_matrix(_FRAME_SHAPE)


def _project_with_cv2(rvec: np.ndarray, tvec: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        points_3d.astype(np.float32),
        rvec.astype(np.float32),
        tvec.astype(np.float32),
        _K.astype(np.float32),
        np.zeros(4, dtype=np.float32),
    )
    return projected.reshape(-1, 2).astype(np.float64)


def _synth_corners(rvec: np.ndarray, tvec: np.ndarray) -> tuple[tuple[float, float], ...]:
    board_corners = np.array([[0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0]], dtype=np.float32)
    points = _project_with_cv2(rvec, tvec, board_corners)
    return tuple((float(p[0]), float(p[1])) for p in points)


def test_camera_pose_roundtrip_recovers_known_pose() -> None:
    # Ground truth: camera tilted 30° down from looking straight north, 10 units above board.
    rvec_gt = np.array([np.radians(-60.0), 0.0, 0.0], dtype=np.float32)  # pitch down
    tvec_gt = np.array([-4.0, -4.0, 12.0], dtype=np.float32)  # place board in front of camera
    corners = _synth_corners(rvec_gt, tvec_gt)

    pose = camera_pose_from_corners(corners, K=_K)

    board_points = np.array([[0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0]], dtype=np.float64)
    recovered_img = project_points(pose, board_points)
    max_err = float(np.max(np.abs(np.array(corners) - recovered_img)))
    assert max_err < 1e-2, f"round-trip error {max_err:.3f} px should be near zero"


def test_project_piece_box_has_8_corners_with_base_and_top() -> None:
    rvec = np.array([np.radians(-45.0), 0.0, 0.0], dtype=np.float32)
    tvec = np.array([-4.0, -4.0, 10.0], dtype=np.float32)
    corners = _synth_corners(rvec, tvec)
    pose = camera_pose_from_corners(corners, K=_K)

    piece_height = 2.0
    box = project_piece_box(pose, row=0, col=0, piece_height=piece_height)

    assert box.shape == (8, 2)
    # Base corners (z=0) project through the same homography as the board corners.
    H = board_to_image_homography(corners)
    expected_base = cv2.perspectiveTransform(
        np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32), H
    ).reshape(4, 2)
    np.testing.assert_allclose(box[:4], expected_base, atol=1.0)
    # Top corners (z=piece_height) project to DIFFERENT pixels.
    assert float(np.min(np.abs(box[4:] - box[:4]))) > 1.0


def test_top_down_camera_produces_tight_piece_bbox() -> None:
    # Camera nearly directly above board (z >> piece height). The residual perspective
    # expansion is only the ratio z_cam / (z_cam - piece_height), which is small for large
    # z_cam; tall and base bboxes should have similar area.
    rvec = np.array([np.pi, 0.0, 0.0], dtype=np.float32)  # flip to look -z
    tvec = np.array([-4.0, -4.0, 200.0], dtype=np.float32)
    corners = _synth_corners(rvec, tvec)
    pose = camera_pose_from_corners(corners, K=_K)

    for row in (0, 4, 7):
        base_box = project_piece_box(pose, row=row, col=3, piece_height=0.0)
        tall_box = project_piece_box(pose, row=row, col=3, piece_height=2.0)
        base_bbox = piece_bbox_from_projection(base_box)
        tall_bbox = piece_bbox_from_projection(tall_box)
        base_area = (base_bbox[2] - base_bbox[0]) * (base_bbox[3] - base_bbox[1])
        tall_area = (tall_bbox[2] - tall_bbox[0]) * (tall_bbox[3] - tall_bbox[1])
        # Nearly top-down: tall_area / base_area should be close to 1.
        ratio = tall_area / base_area
        assert ratio < 1.15, f"row {row}: top-down ratio {ratio:.3f} is too large"


def test_oblique_camera_piece_bbox_grows_with_height() -> None:
    # Chesscog-ish oblique camera (~45°). Taller pieces project to much taller bboxes.
    rvec = np.array([np.radians(-45.0), 0.0, 0.0], dtype=np.float32)
    tvec = np.array([-4.0, -4.0, 10.0], dtype=np.float32)
    corners = _synth_corners(rvec, tvec)
    pose = camera_pose_from_corners(corners, K=_K)

    short_box = project_piece_box(pose, row=0, col=3, piece_height=0.5)
    tall_box = project_piece_box(pose, row=0, col=3, piece_height=2.5)
    short_bbox = piece_bbox_from_projection(short_box)
    tall_bbox = piece_bbox_from_projection(tall_box)
    short_h = short_bbox[3] - short_bbox[1]
    tall_h = tall_bbox[3] - tall_bbox[1]
    # Taller piece -> taller image bbox, and proportional to the increased height.
    assert tall_h > short_h * 2.0, f"oblique tall_h {tall_h:.0f} not >> short_h {short_h:.0f}"


def _camera_rvec_tvec(
    camera_pos_world: np.ndarray, look_at_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build rvec/tvec for a camera at ``camera_pos_world`` looking at ``look_at_world``.

    Assumes no-roll (world +z stays "up" as much as possible in the image).
    """
    forward = look_at_world - camera_pos_world
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    cam_down = np.cross(forward, right)
    cam_down = cam_down / np.linalg.norm(cam_down)
    # Standard CV camera: x=right, y=down, z=forward. Rows of R_w_c are these in world coords.
    R_w_c = np.stack([right, cam_down, forward], axis=0)
    rvec, _ = cv2.Rodrigues(R_w_c)
    tvec = -R_w_c @ camera_pos_world
    return rvec.flatten().astype(np.float32), tvec.astype(np.float32)


def test_chesscog_angle_produces_chesscog_like_top_extension() -> None:
    """At a chesscog-like oblique camera angle, a piece height-2.5 projection gives top_ext~3.

    For a camera at ``(4, 5, 5)`` (elevation ~45 deg, distance ~5 from board center), a piece
    of height 2.5 at row=0 back-projects its top through the board-plane homography to
    approximately row=-2.5 (i.e. 3 squares above the piece's base-top edge, which lies at
    row=0). That matches chesscog's ``top_ext_max = 3.0``.

    Rationale: a physically-realistic king in chesscog's chess sets is ~2.5 board units tall.
    At their camera angle, this maps to their empirical top_ext=3. Different camera angles
    change the number automatically via the projection -- that's the whole point.
    """
    cam_pos = np.array([4.0, 5.0, 5.0])
    look_at = np.array([4.0, 4.0, 0.0])
    rvec, tvec = _camera_rvec_tvec(cam_pos, look_at)
    corners = _synth_corners(rvec, tvec)
    pose = camera_pose_from_corners(corners, K=_K)

    box = project_piece_box(pose, row=0, col=3, piece_height=2.5)
    top_center = box[4:].mean(axis=0)
    H = board_to_image_homography(corners)
    Hinv = np.linalg.inv(H)
    homogeneous = Hinv @ np.array([top_center[0], top_center[1], 1.0])
    board_top = homogeneous[:2] / homogeneous[2]

    # Expected: top_ext ~ 3 squares (chesscog's top_ext_max). Back-projected row should
    # be roughly -2.5 (piece-center row 0.5 minus top_ext 3 = -2.5).
    assert -3.5 < board_top[1] < -1.5, (
        f"chesscog-angle top back-projects to row {board_top[1]:.2f} "
        f"(expected roughly -2.5 for top_ext=3)"
    )


def test_oblique_vs_top_down_produces_very_different_top_extensions() -> None:
    """The whole point of the projected crop: top extension VARIES with camera angle."""

    def top_extension(cam_pos: np.ndarray) -> float:
        rvec, tvec = _camera_rvec_tvec(cam_pos, np.array([4.0, 4.0, 0.0]))
        corners = _synth_corners(rvec, tvec)
        pose = camera_pose_from_corners(corners, K=_K)
        box = project_piece_box(pose, row=0, col=3, piece_height=2.0)
        top_center = box[4:].mean(axis=0)
        H = board_to_image_homography(corners)
        Hinv = np.linalg.inv(H)
        hom = Hinv @ np.array([top_center[0], top_center[1], 1.0])
        board_top_y = hom[1] / hom[2]
        # top_ext = how far above the piece base row the top back-projects to.
        # base center is at row 0.5, so top_ext = 0.5 - board_top_y.
        return 0.5 - float(board_top_y)

    oblique = top_extension(np.array([4.0, 5.0, 5.0]))  # 45 deg
    closer_to_top_down = top_extension(np.array([4.0, 4.5, 20.0]))  # much higher

    assert oblique > 2.0, f"chesscog-like oblique top_ext {oblique:.2f} should be >2"
    assert closer_to_top_down < 1.0, (
        f"closer-to-top-down top_ext {closer_to_top_down:.2f} should be <1"
    )
    assert oblique > closer_to_top_down + 1.5


def test_project_piece_bboxes_returns_row_major_piece_regions() -> None:
    rvec = np.array([np.radians(-45.0), 0.0, 0.0], dtype=np.float32)
    tvec = np.array([-4.0, -4.0, 10.0], dtype=np.float32)
    corners = _synth_corners(rvec, tvec)

    bboxes = project_piece_bboxes(corners, frame_shape=_FRAME_SHAPE)

    assert bboxes.shape == (64, 4)
    square_bbox = square_bbox_from_corners(corners, row=0, col=0)
    piece_bbox = bboxes[0]
    assert piece_bbox[3] - piece_bbox[1] > square_bbox[3] - square_bbox[1]


def test_extract_board_neighborhood_crop_returns_relative_corners() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    corners = ((20.0, 10.0), (100.0, 10.0), (100.0, 90.0), (20.0, 90.0))

    crop = extract_board_neighborhood_crop(image, corners, crop_margin=0.25)

    assert crop.image_bgr.shape == (100, 120, 3)
    np.testing.assert_allclose(
        crop.corners,
        np.array([[20.0, 10.0], [100.0, 10.0], [100.0, 90.0], [20.0, 90.0]], dtype=np.float32),
    )


def test_transform_projected_bboxes_to_crop_space_differs_from_reprojecting_in_crop_space() -> None:
    rvec = np.array([np.radians(-45.0), 0.0, 0.0], dtype=np.float32)
    tvec = np.array([-4.0, -4.0, 10.0], dtype=np.float32)
    corners = _synth_corners(rvec, tvec)
    image = np.zeros(_FRAME_SHAPE, dtype=np.uint8)

    crop = extract_board_neighborhood_crop(image, corners, crop_margin=0.18)
    transformed_bboxes = transform_projected_bboxes_to_crop_space(
        project_piece_bboxes(corners, frame_shape=image.shape),
        crop,
        output_shape=224,
    )

    scaled_corners = crop.corners.copy()
    scaled_corners[:, 0] *= 224.0 / crop.image_bgr.shape[1]
    scaled_corners[:, 1] *= 224.0 / crop.image_bgr.shape[0]
    reprojected_bboxes = project_piece_bboxes(scaled_corners.tolist(), frame_shape=(224, 224, 3))

    c8_index = 2
    assert not np.allclose(transformed_bboxes[c8_index], reprojected_bboxes[c8_index], atol=1e-3)


def test_project_square_base_quad_matches_axis_aligned_board_geometry() -> None:
    corners = ((20.0, 10.0), (100.0, 10.0), (100.0, 90.0), (20.0, 90.0))

    quad = project_square_base_quad(corners, row=2, col=3)

    np.testing.assert_allclose(
        quad,
        np.array([[50.0, 30.0], [60.0, 30.0], [60.0, 40.0], [50.0, 40.0]], dtype=np.float64),
        atol=1e-6,
    )


def test_square_bbox_from_corners_supports_padding() -> None:
    corners = ((20.0, 10.0), (100.0, 10.0), (100.0, 90.0), (20.0, 90.0))

    bbox = square_bbox_from_corners(corners, row=2, col=3, pad_ratio=0.1)

    np.testing.assert_allclose(bbox, (49.0, 29.0, 61.0, 41.0), atol=1e-6)


def test_default_piece_height_roundtrips_to_expected_range() -> None:
    """Check that DEFAULT_PIECE_HEIGHT is in the sensible chess-piece range."""
    assert 1.0 <= DEFAULT_PIECE_HEIGHT <= 3.0
