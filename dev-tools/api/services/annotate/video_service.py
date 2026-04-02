"""Service layer for video annotation, wrapping overlay pipeline modules."""

import base64
import collections
import os
import threading
import uuid
from typing import Any

import chess
import cv2
import numpy as np

_FRAME_CACHE_MAX = 64

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
        "lock": threading.Lock(),
        "frame_cache": collections.OrderedDict(),
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

    total = session["total_frames"]
    frame_index = max(0, min(frame_index, total - 1))

    cache = session["frame_cache"]

    # Fast path: return cached frame without acquiring the lock
    if frame_index in cache:
        cache.move_to_end(frame_index)
        return cache[frame_index]

    with session["lock"]:
        # Double-check after acquiring lock
        if frame_index in cache:
            cache.move_to_end(frame_index)
            return cache[frame_index]

        cap = session["cap"]
        # Try the exact frame first, then fall back to nearby frames
        for offset in [0, -1, -5, -10]:
            idx = max(0, frame_index + offset)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpeg_bytes = buffer.tobytes()
                cache[frame_index] = jpeg_bytes
                if len(cache) > _FRAME_CACHE_MAX:
                    cache.popitem(last=False)
                return jpeg_bytes

    raise ValueError(f"Cannot read frame {frame_index}")


def _encode_crop_b64(frame: np.ndarray, bbox: tuple) -> str:
    x, y, w, h = bbox
    crop = frame[y : y + h, x : x + w]
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _get_clip_calibration(clip_id: int):
    """Build a LayoutCalibration from a video_clips DB row."""
    from pipeline.overlay.calibration import LayoutCalibration
    from api.services.videos.crawl_service import get_video_clip

    clip = get_video_clip(clip_id)
    if clip is None:
        raise ValueError(f"Clip {clip_id} not found")
    return clip, LayoutCalibration(
        overlay=tuple(clip["overlay_bbox"]),
        camera=tuple(clip["camera_bbox"]),
        ref_resolution=tuple(clip["ref_resolution"]),
        board_flipped=clip["board_flipped"],
        board_theme=clip["board_theme"],
    )


def read_overlay_at_frame(session_id: str, frame_index: int, clip_id: int | None = None) -> dict:
    """Read overlay FEN at a specific frame. Optionally use a clip's calibration."""
    from pipeline.overlay.grid_detector import detect_grid
    from pipeline.overlay.piece_classifier import read_fen_with_grid

    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    if clip_id is not None:
        _, calibration = _get_clip_calibration(clip_id)
    else:
        calibration = session.get("calibration")
    if calibration is None:
        raise ValueError("No calibration for this session")

    fps = session["fps"]
    with session["lock"]:
        cap = session["cap"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_index}")

    overlay_crop_b64 = _encode_crop_b64(frame, calibration.overlay)
    camera_crop_b64 = _encode_crop_b64(frame, calibration.camera)

    # Read FEN using DINOv2 piece classifier with auto grid detection
    ox, oy, ow, oh = calibration.overlay
    overlay_img = frame[oy : oy + oh, ox : ox + ow]

    grid = detect_grid(overlay_img)
    fen: str | None = None
    board: chess.Board | None = None
    if grid is not None:
        fen = read_fen_with_grid(overlay_img, grid)
        board = chess.Board(fen=None)
        board.set_fen(fen + " w - - 0 1")

    return {
        "frame_index": frame_index,
        "timestamp_seconds": round(frame_index / fps, 3) if fps > 0 else 0,
        "fen": fen,
        "board_ascii": str(board) if board else None,
        "overlay_crop_b64": overlay_crop_b64,
        "camera_crop_b64": camera_crop_b64,
    }


def detect_moves(session_id: str, sample_fps: float = 2.0, clip_id: int | None = None) -> dict:
    """Run full move detection on the video. If clip_id is given, restrict to that clip's time range and calibration."""
    from pipeline.overlay.overlay_move_detector import (
        detect_moves as run_detect,
    )
    from pipeline.overlay.grid_detector import detect_grid
    from pipeline.overlay.piece_classifier import read_fen_with_grid

    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    clip_data = None
    if clip_id is not None:
        clip_data, calibration = _get_clip_calibration(clip_id)
    else:
        calibration = session.get("calibration")
    if calibration is None:
        raise ValueError("No calibration for this session")

    cap = session["cap"]
    fps = session["fps"]
    total_frames = session["total_frames"]

    # Determine frame range
    start_frame = 0
    end_frame = total_frames
    start_time = 0.0
    if clip_data is not None:
        start_frame = int(clip_data["start_time"] * fps) if fps > 0 else 0
        if clip_data["end_time"] is not None:
            end_frame = int(clip_data["end_time"] * fps)
        start_time = clip_data["start_time"]

    # Sample frames
    frame_interval = max(1, int(fps / sample_fps))

    fens: list[str | None] = []
    frame_indices: list[int] = []
    num_readable = 0

    lock = session["lock"]
    for idx in range(start_frame, end_frame, frame_interval):
        with lock:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
        if not ret:
            fens.append(None)
            frame_indices.append(idx)
            continue

        ox, oy, ow, oh = calibration.overlay
        overlay_img = frame[oy : oy + oh, ox : ox + ow]
        grid = detect_grid(overlay_img)
        fen = read_fen_with_grid(overlay_img, grid) if grid else None
        fens.append(fen)
        frame_indices.append(idx)
        if fen is not None:
            num_readable += 1

    # Detect moves from FEN sequence
    segments = run_detect(
        fens=fens,
        frame_indices=frame_indices,
        fps=fps,
        start_time=start_time,
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
                "confidence": round(m.confidence, 3),
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


def generate_clips(session_id: str, clip_id: int | None = None) -> dict:
    """Generate training clips from the video. If clip_id given, restrict to that clip."""
    from pipeline.overlay.overlay_clip_generator import OverlayClipGenerator

    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    clip_data = None
    if clip_id is not None:
        clip_data, calibration = _get_clip_calibration(clip_id)
    else:
        calibration = session.get("calibration")
    if calibration is None:
        raise ValueError("No calibration for this session — calibrate first")

    video_path = session["video_path"]
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Build suffix for clip-specific output
    output_suffix = ""
    start_time = None
    end_time = None
    if clip_data is not None:
        output_suffix = f"_clip{clip_data['clip_index']}"
        start_time = clip_data["start_time"]
        end_time = clip_data["end_time"]

    generator = OverlayClipGenerator()
    results = generator.generate_clips(
        video_path, calibration, video_id=video_id + output_suffix,
        start_time=start_time, end_time=end_time,
    )

    clips = []
    for r in results:
        clips.append({
            "filepath": r["filepath"],
            "num_frames": r["num_frames"],
            "num_moves": r["num_moves"],
            "game_index": r["game_index"],
            "pgn_moves": r.get("pgn_moves", ""),
        })

    return {"clips": clips, "total_clips": len(clips)}


def delete_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        session["cap"].release()
