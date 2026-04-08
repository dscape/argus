"""Service layer wrapping pipeline.overlay.calibration."""

from pipeline.overlay.calibration import (
    LayoutCalibration,
    get_calibration,
    list_calibrations,
    load_config,
    save_config,
    set_calibration,
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


def propose(channel_handle: str, video_id: str | None = None) -> dict | None:
    """Auto-propose calibration from screening data.

    Uses sampled video frames to detect the overlay, OTB board, theme, and
    orientation. If video_id is given, proposes from that single video;
    otherwise aggregates proposals from multiple approved videos on the channel.
    """
    from pipeline.overlay.auto_calibration import (
        propose_calibration,
        propose_calibration_for_channel,
    )

    if video_id:
        proposal = propose_calibration(video_id)
    else:
        proposal = propose_calibration_for_channel(channel_handle)

    if proposal is None:
        return None

    return {
        "channel_handle": channel_handle,
        "overlay": list(proposal.overlay),
        "camera": list(proposal.camera),
        "ref_resolution": list(proposal.ref_resolution),
        "board_flipped": proposal.board_flipped,
        "board_theme": proposal.board_theme,
        "theme_confidence": round(proposal.theme_confidence, 3),
        "orientation_confidence": round(proposal.orientation_confidence, 3),
    }
