"""Service layer for video annotation, wrapping overlay pipeline modules."""

import base64
import uuid
from typing import Any

import cv2
import numpy as np

# Session storage
_sessions: dict[str, dict[str, Any]] = {}


def open_video(video_path: str, channel_handle: str | None = None) -> dict:
    """Open a video file and create a session."""
    from pipeline.overlay.calibration import get_calibration

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    calibration = None
    calibration_dict = None
    if channel_handle:
        cal = get_calibration(channel_handle)
        if cal:
            calibration = cal.scale_to_resolution(width, height)
            calibration_dict = {
                "channel_handle": channel_handle,
                "overlay": list(calibration.overlay),
                "camera": list(calibration.camera),
                "ref_resolution": [width, height],
                "board_flipped": calibration.board_flipped,
                "board_theme": calibration.board_theme,
            }

    session_id = str(uuid.uuid4())[:8]
    _sessions[session_id] = {
        "cap": cap,
        "calibration": calibration,
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
    }

    return {
        "session_id": session_id,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
        "width": width,
        "height": height,
        "has_calibration": calibration is not None,
        "calibration": calibration_dict,
    }


def get_frame_jpeg(session_id: str, frame_index: int) -> bytes:
    """Read a specific frame from the video as JPEG bytes."""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    cap = session["cap"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_index}")

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def _encode_crop_b64(frame: np.ndarray, bbox: tuple) -> str:
    x, y, w, h = bbox
    crop = frame[y : y + h, x : x + w]
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def read_overlay_at_frame(session_id: str, frame_index: int) -> dict:
    """Read overlay FEN at a specific frame."""
    from pipeline.overlay.overlay_reader import OverlayReader

    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    calibration = session.get("calibration")
    if calibration is None:
        raise ValueError("No calibration for this session")

    cap = session["cap"]
    fps = session["fps"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_index}")

    overlay_crop_b64 = _encode_crop_b64(frame, calibration.overlay)
    camera_crop_b64 = _encode_crop_b64(frame, calibration.camera)

    # Read FEN
    ox, oy, ow, oh = calibration.overlay
    overlay_img = frame[oy : oy + oh, ox : ox + ow]

    reader = OverlayReader(board_theme=calibration.board_theme)
    board = reader.read_board(overlay_img, flipped=calibration.board_flipped)

    return {
        "frame_index": frame_index,
        "timestamp_seconds": round(frame_index / fps, 3) if fps > 0 else 0,
        "fen": board.board_fen() if board else None,
        "board_ascii": str(board) if board else None,
        "overlay_crop_b64": overlay_crop_b64,
        "camera_crop_b64": camera_crop_b64,
    }


def detect_moves(session_id: str, sample_fps: float = 2.0) -> dict:
    """Run full move detection on the video."""
    from pipeline.overlay.overlay_move_detector import (
        detect_moves as run_detect,
    )
    from pipeline.overlay.overlay_reader import OverlayReader

    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    calibration = session.get("calibration")
    if calibration is None:
        raise ValueError("No calibration for this session")

    cap = session["cap"]
    fps = session["fps"]
    total_frames = session["total_frames"]

    # Sample frames
    frame_interval = max(1, int(fps / sample_fps))
    reader = OverlayReader(board_theme=calibration.board_theme)

    fens: list[str | None] = []
    frame_indices: list[int] = []
    num_readable = 0

    for idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            fens.append(None)
            frame_indices.append(idx)
            continue

        ox, oy, ow, oh = calibration.overlay
        overlay_img = frame[oy : oy + oh, ox : ox + ow]
        fen = reader.read_fen(overlay_img, flipped=calibration.board_flipped)
        fens.append(fen)
        frame_indices.append(idx)
        if fen is not None:
            num_readable += 1

    # Detect moves from FEN sequence
    segments = run_detect(
        fens=fens,
        frame_indices=frame_indices,
        fps=fps,
        start_time=0.0,
    )

    result_segments = []
    for i, seg in enumerate(segments):
        seg_moves = []
        for m in seg.moves:
            seg_moves.append({
                "move_index": m.move_index,
                "move_uci": m.move_uci,
                "move_san": m.move_san,
                "frame_idx": m.frame_idx,
                "timestamp_seconds": round(m.timestamp_seconds, 3),
                "fen_before": m.fen_before,
                "fen_after": m.fen_after,
            })
        result_segments.append({
            "game_index": i,
            "num_moves": len(seg.moves),
            "pgn_moves": seg.pgn_moves,
            "moves": seg_moves,
            "start_frame": seg.moves[0].frame_idx if seg.moves else 0,
            "end_frame": seg.moves[-1].frame_idx if seg.moves else 0,
        })

    return {
        "num_frames_sampled": len(fens),
        "num_readable": num_readable,
        "segments": result_segments,
    }


def delete_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        session["cap"].release()
