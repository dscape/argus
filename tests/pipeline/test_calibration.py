from pipeline.overlay.calibration import (
    LayoutCalibration,
    bbox_area_ratio,
    calibration_has_usable_camera_crop,
    calibration_is_usable,
    is_camera_bbox_usable,
    is_overlay_bbox_usable,
    is_placeholder_bbox,
)


def test_placeholder_bbox_is_not_usable_camera_crop():
    assert is_placeholder_bbox((0, 0, 100, 100)) is True
    assert is_camera_bbox_usable((0, 0, 100, 100), (1920, 1080)) is False


def test_tiny_bbox_is_not_usable_camera_crop():
    bbox = (0, 0, 132, 132)
    assert bbox_area_ratio(bbox, (1920, 1080)) < 0.02
    assert is_camera_bbox_usable(bbox, (1920, 1080)) is False


def test_large_panel_bbox_is_not_usable_camera_crop():
    bbox = (1079, 0, 841, 1080)
    assert bbox_area_ratio(bbox, (1920, 1080)) > 0.25
    assert is_camera_bbox_usable(bbox, (1920, 1080)) is False


def test_tight_board_bbox_is_usable_camera_crop():
    bbox = (1294, 894, 499, 178)
    assert is_camera_bbox_usable(bbox, (1920, 1080)) is True


def test_calibration_is_usable_requires_valid_overlay_and_camera():
    usable = LayoutCalibration(
        overlay=(56, 10, 1056, 1056),
        camera=(1294, 894, 499, 178),
        ref_resolution=(1920, 1080),
    )
    invalid_camera = LayoutCalibration(
        overlay=(56, 10, 1056, 1056),
        camera=(1079, 0, 841, 1080),
        ref_resolution=(1920, 1080),
    )

    assert is_overlay_bbox_usable(usable.overlay, usable.ref_resolution) is True
    assert calibration_has_usable_camera_crop(usable) is True
    assert calibration_is_usable(usable) is True
    assert calibration_is_usable(invalid_camera) is False
