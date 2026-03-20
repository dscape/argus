"""Service layer wrapping pipeline.overlay.calibration."""

from pipeline.overlay.calibration import (
    LayoutCalibration,
    get_calibration,
    set_calibration,
    list_calibrations,
    load_config,
    save_config,
    CONFIG_PATH,
)


def list_all() -> list[dict]:
    """Return all calibrations as dicts."""
    cals = list_calibrations()
    result = []
    for handle, cal in cals.items():
        result.append({
            "channel_handle": handle,
            "overlay": list(cal.overlay),
            "camera": list(cal.camera),
            "ref_resolution": list(cal.ref_resolution),
            "board_flipped": cal.board_flipped,
            "board_theme": cal.board_theme,
        })
    return result


def get_one(channel_handle: str) -> dict | None:
    cal = get_calibration(channel_handle)
    if cal is None:
        return None
    return {
        "channel_handle": channel_handle,
        "overlay": list(cal.overlay),
        "camera": list(cal.camera),
        "ref_resolution": list(cal.ref_resolution),
        "board_flipped": cal.board_flipped,
        "board_theme": cal.board_theme,
    }


def save(channel_handle: str, data: dict) -> dict:
    cal = LayoutCalibration(
        overlay=tuple(data["overlay"]),
        camera=tuple(data["camera"]),
        ref_resolution=tuple(data.get("ref_resolution", [1920, 1080])),
        board_flipped=data.get("board_flipped", False),
        board_theme=data.get("board_theme", "lichess_default"),
    )
    set_calibration(channel_handle, cal)
    return {
        "channel_handle": channel_handle,
        "overlay": list(cal.overlay),
        "camera": list(cal.camera),
        "ref_resolution": list(cal.ref_resolution),
        "board_flipped": cal.board_flipped,
        "board_theme": cal.board_theme,
    }


def delete(channel_handle: str) -> bool:
    data = load_config()
    layouts = data.get("layouts", {})
    if channel_handle not in layouts:
        return False
    del layouts[channel_handle]
    data["layouts"] = layouts
    save_config(data)
    return True
