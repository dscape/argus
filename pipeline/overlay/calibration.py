"""Per-channel layout calibration for OBS overlay videos.

Stores and retrieves crop coordinates for the 2D overlay region and OTB camera
region. Calibrations are stored in a YAML config file and are static per channel
(OBS layouts don't change between videos on the same channel).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join("configs", "pipeline", "overlay_layouts.yaml")


@dataclass
class LayoutCalibration:
    """Crop coordinates for a channel's OBS layout."""

    overlay: tuple[int, int, int, int]  # x, y, w, h
    camera: tuple[int, int, int, int]  # x, y, w, h
    ref_resolution: tuple[int, int]  # width, height
    board_flipped: bool = False
    board_theme: str = "lichess_default"
    move_delay_seconds: float = 2.0  # Broadcast delay: OTB move happens before overlay updates

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
