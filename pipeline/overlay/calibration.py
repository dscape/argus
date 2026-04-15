"""Per-channel layout calibration for OBS overlay videos.

Stores and retrieves crop coordinates for the 2D overlay region and OTB camera
region. Calibrations are stored in a YAML config file and are static per channel
(OBS layouts don't change between videos on the same channel).
"""

import logging
import os
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join("configs", "annotate", "overlay_layouts.yaml")

# Common board themes: hex colors for light/dark squares.
BOARD_THEMES: dict[str, dict[str, str]] = {
    "lichess_default": {"light": "#F0D9B5", "dark": "#B58863"},
    "chess_com_green": {"light": "#EEEED2", "dark": "#769656"},
    "chess_com_brown": {"light": "#F0D9B5", "dark": "#B58863"},
}

_PLACEHOLDER_BBOX = (0, 0, 100, 100)
_MIN_CAMERA_BBOX_AREA_RATIO = 0.02
_MAX_CAMERA_BBOX_AREA_RATIO = 0.25


def _normalize_bbox(bbox: tuple[int, int, int, int] | list[int]) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in bbox)


def is_placeholder_bbox(bbox: tuple[int, int, int, int] | list[int]) -> bool:
    return _normalize_bbox(bbox) == _PLACEHOLDER_BBOX


def bbox_area_ratio(
    bbox: tuple[int, int, int, int] | list[int],
    ref_resolution: tuple[int, int] | list[int],
) -> float:
    _, _, width, height = _normalize_bbox(bbox)
    ref_width, ref_height = (int(v) for v in ref_resolution)
    if ref_width <= 0 or ref_height <= 0:
        return 0.0
    return (width * height) / (ref_width * ref_height)


def is_bbox_within_frame(
    bbox: tuple[int, int, int, int] | list[int],
    ref_resolution: tuple[int, int] | list[int],
) -> bool:
    x, y, width, height = _normalize_bbox(bbox)
    ref_width, ref_height = (int(v) for v in ref_resolution)
    if width <= 0 or height <= 0 or ref_width <= 0 or ref_height <= 0:
        return False
    if x < 0 or y < 0:
        return False
    return x + width <= ref_width and y + height <= ref_height


def is_overlay_bbox_usable(
    bbox: tuple[int, int, int, int] | list[int],
    ref_resolution: tuple[int, int] | list[int],
) -> bool:
    return not is_placeholder_bbox(bbox) and is_bbox_within_frame(bbox, ref_resolution)


def is_camera_bbox_usable(
    bbox: tuple[int, int, int, int] | list[int],
    ref_resolution: tuple[int, int] | list[int],
    *,
    min_area_ratio: float = _MIN_CAMERA_BBOX_AREA_RATIO,
    max_area_ratio: float = _MAX_CAMERA_BBOX_AREA_RATIO,
) -> bool:
    if is_placeholder_bbox(bbox):
        return False
    if not is_bbox_within_frame(bbox, ref_resolution):
        return False
    area_ratio = bbox_area_ratio(bbox, ref_resolution)
    return min_area_ratio <= area_ratio <= max_area_ratio


def calibration_has_usable_camera_crop(calibration: "LayoutCalibration") -> bool:
    return is_camera_bbox_usable(calibration.camera, calibration.ref_resolution)


def calibration_is_usable(calibration: "LayoutCalibration") -> bool:
    return is_overlay_bbox_usable(
        calibration.overlay,
        calibration.ref_resolution,
    ) and calibration_has_usable_camera_crop(calibration)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


@dataclass
class LayoutCalibration:
    """Crop coordinates for a channel's OBS layout."""

    overlay: tuple[int, int, int, int]  # x, y, w, h
    camera: tuple[int, int, int, int]  # x, y, w, h
    ref_resolution: tuple[int, int]  # width, height
    board_flipped: bool = False
    board_theme: str = "lichess_default"
    move_delay_seconds: float = 2.0  # Broadcast delay: OTB move happens before overlay updates
    # Used for estimated OTB timing metadata; training labels stay on overlay-confirm frames.

    def scale_to_resolution(self, width: int, height: int) -> "LayoutCalibration":
        """Return a new calibration scaled to a different resolution."""
        ref_w, ref_h = self.ref_resolution

        if ref_w == width and ref_h == height:
            return self

        sx = width / ref_w
        sy = height / ref_h

        def scale_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            x, y, w, h = bbox
            return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

        return LayoutCalibration(
            overlay=scale_bbox(self.overlay),
            camera=scale_bbox(self.camera),
            ref_resolution=(width, height),
            board_flipped=self.board_flipped,
            board_theme=self.board_theme,
            move_delay_seconds=self.move_delay_seconds,
        )


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load the overlay layouts config file."""
    if not os.path.exists(config_path):
        return {"layouts": {}}

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return data


def save_config(data: dict, config_path: str = CONFIG_PATH):
    """Save the overlay layouts config file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_calibration(
    channel_handle: str,
    config_path: str = CONFIG_PATH,
) -> LayoutCalibration | None:
    """Load calibration for a channel from the config file."""
    data = load_config(config_path)
    layouts = data.get("layouts", {})

    entry = layouts.get(channel_handle)
    if entry is None:
        return None

    return LayoutCalibration(
        overlay=tuple(entry["overlay"]),
        camera=tuple(entry["camera"]),
        ref_resolution=tuple(entry.get("ref_resolution", [1920, 1080])),
        board_flipped=entry.get("board_flipped", False),
        board_theme=entry.get("board_theme", "lichess_default"),
        move_delay_seconds=entry.get("move_delay_seconds", 2.0),
    )


def set_calibration(
    channel_handle: str,
    calibration: LayoutCalibration,
    config_path: str = CONFIG_PATH,
):
    """Save calibration for a channel to the config file."""
    data = load_config(config_path)
    if "layouts" not in data:
        data["layouts"] = {}

    data["layouts"][channel_handle] = {
        "overlay": list(calibration.overlay),
        "camera": list(calibration.camera),
        "ref_resolution": list(calibration.ref_resolution),
        "board_flipped": calibration.board_flipped,
        "board_theme": calibration.board_theme,
        "move_delay_seconds": calibration.move_delay_seconds,
    }

    save_config(data, config_path)
    logger.info(f"Saved calibration for {channel_handle}")


def list_calibrations(config_path: str = CONFIG_PATH) -> dict[str, LayoutCalibration]:
    """List all stored calibrations."""
    data = load_config(config_path)
    result = {}
    for handle, entry in data.get("layouts", {}).items():
        result[handle] = LayoutCalibration(
            overlay=tuple(entry["overlay"]),
            camera=tuple(entry["camera"]),
            ref_resolution=tuple(entry.get("ref_resolution", [1920, 1080])),
            board_flipped=entry.get("board_flipped", False),
            board_theme=entry.get("board_theme", "lichess_default"),
            move_delay_seconds=entry.get("move_delay_seconds", 2.0),
        )
    return result
