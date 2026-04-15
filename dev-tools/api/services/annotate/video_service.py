"""Service layer for video annotation, wrapping overlay pipeline modules."""

import base64
import collections
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np

from pipeline.analysis.board_reading import read_overlay_crop
from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.overlay.board_crop import find_board_grid_in_crop, find_stable_board_grid
from pipeline.overlay.piece_classifier import read_board_with_grid
from pipeline.overlay.sequence_reader import LockedOverlaySequenceReader

_FRAME_CACHE_MAX = 64
logger = logging.getLogger(__name__)

# ── Job persistence ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_JOBS_FILE = _PROJECT_ROOT / "data" / ".job_state" / "move_detection_jobs.json"
_jobs_lock = threading.Lock()
_PERSIST_INTERVAL = 2.0  # seconds – debounce for progress ticks
_last_persist_time: float = 0.0

# Session storage
_sessions: dict[str, dict[str, Any]] = {}
_move_detection_jobs: dict[str, dict[str, Any]] = {}


def _persist_jobs(force: bool = False) -> None:
    """Write the jobs dict to disk. Caller MUST hold _jobs_lock."""
    global _last_persist_time
    now = time.monotonic()
    if not force and (now - _last_persist_time) < _PERSIST_INTERVAL:
        return
    try:
        _JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _JOBS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_move_detection_jobs, default=str))
        os.replace(str(tmp), str(_JOBS_FILE))
        _last_persist_time = now
    except OSError:
        logger.warning("Failed to persist job state to %s", _JOBS_FILE, exc_info=True)


def _load_jobs() -> None:
    """Restore jobs from disk on startup; mark orphaned running jobs as failed."""
    global _move_detection_jobs
    if not _JOBS_FILE.exists():
        return
    try:
        data = json.loads(_JOBS_FILE.read_text())
        for job in data.values():
            if job.get("status") == "running":
                job["status"] = "failed"
                job["error"] = "Server restarted while job was running"
        _move_detection_jobs.update(data)
        logger.info("Restored %d job(s) from %s", len(data), _JOBS_FILE)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not load job state from %s", _JOBS_FILE, exc_info=True)


_load_jobs()


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
        "clip_grid_cache": {},
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


def _read_overlay_crop_frame(
    session: dict[str, Any],
    calibration,
    frame_index: int,
) -> np.ndarray | None:
    total_frames = int(session["total_frames"])
    frame_index = max(0, min(int(frame_index), total_frames - 1))

    with session["lock"]:
        cap = session["cap"]
        for offset in [0, -1, -5, -10]:
            current_index = max(0, frame_index + offset)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            ox, oy, ow, oh = calibration.overlay
            return frame[oy : oy + oh, ox : ox + ow]
    return None


def _clip_grid_candidate_indices(
    clip_data: dict[str, Any],
    *,
    fps: float,
    total_frames: int,
    preferred_frame_index: int | None = None,
) -> list[int]:
    start_frame = max(0, int(round(float(clip_data["start_time"]) * fps)))
    raw_end_time = clip_data.get("end_time")
    if raw_end_time is None:
        end_frame = max(start_frame, total_frames - 1)
    else:
        end_frame = max(start_frame, min(total_frames - 1, int(round(float(raw_end_time) * fps))))

    midpoint = start_frame + max(0, end_frame - start_frame) // 2
    quarter = start_frame + max(0, end_frame - start_frame) // 4
    three_quarter = start_frame + (3 * max(0, end_frame - start_frame)) // 4

    candidates = [
        preferred_frame_index,
        midpoint,
        quarter,
        three_quarter,
        start_frame,
        end_frame,
    ]

    result: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        clamped = max(0, min(int(candidate), total_frames - 1))
        if clamped in seen:
            continue
        seen.add(clamped)
        result.append(clamped)
    return result


def _get_or_init_clip_grid(
    session: dict[str, Any],
    clip_data: dict[str, Any],
    calibration,
    *,
    preferred_frame_index: int | None = None,
):
    cache = session.setdefault("clip_grid_cache", {})
    clip_id = int(clip_data["id"])
    cached = cache.get(clip_id)
    if cached is not None:
        return cached

    candidate_indices = _clip_grid_candidate_indices(
        clip_data,
        fps=float(session["fps"]),
        total_frames=int(session["total_frames"]),
        preferred_frame_index=preferred_frame_index,
    )
    grid = find_stable_board_grid(
        lambda index: _read_overlay_crop_frame(session, calibration, index),
        candidate_indices,
    )
    if grid is not None:
        cache[clip_id] = grid
    return grid


def read_overlay_at_frame(
    session_id: str,
    frame_index: int,
    clip_id: int | None = None,
    reader_backend: str = "overlay",
) -> dict:
    """Read overlay FEN at a specific frame. Optionally use a clip's calibration."""

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

    fps = session["fps"]
    with session["lock"]:
        cap = session["cap"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_index}")

    overlay_crop_b64 = _encode_crop_b64(frame, calibration.overlay)
    camera_crop_b64 = _encode_crop_b64(frame, calibration.camera)

    ox, oy, ow, oh = calibration.overlay
    overlay_img = frame[oy : oy + oh, ox : ox + ow]

    config = VideoAnalysisConfig(reader_backend=reader_backend, scene_backend="none", device="cpu")
    read_result = None
    if reader_backend == "overlay" and clip_data is not None:
        locked_grid = _get_or_init_clip_grid(
            session,
            clip_data,
            calibration,
            preferred_frame_index=frame_index,
        )
        if locked_grid is not None:
            board_state = read_board_with_grid(overlay_img, locked_grid, device=config.device)
            read_result = {
                "fen": board_state.fen,
                "method": "overlay_locked_grid",
            }

    if read_result is None:
        crop_result = read_overlay_crop(overlay_img, config)
        read_result = {
            "fen": crop_result.fen,
            "method": crop_result.method,
        }

    fen = read_result["fen"]
    board: chess.Board | None = None
    if fen is not None:
        board = chess.Board(fen=None)
        board.set_fen(fen + " w - - 0 1")

    return {
        "frame_index": frame_index,
        "timestamp_seconds": round(frame_index / fps, 3) if fps > 0 else 0,
        "fen": fen,
        "board_ascii": str(board) if board else None,
        "read_method": read_result["method"],
        "overlay_crop_b64": overlay_crop_b64,
        "camera_crop_b64": camera_crop_b64,
    }


def detect_moves(
    session_id: str,
    sample_fps: float = 2.0,
    clip_id: int | None = None,
    reader_backend: str = "overlay",
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> dict:
    """Run full move detection on the video. If clip_id is given, restrict to that clip's time range and calibration."""
    from pipeline.overlay.overlay_move_detector import detect_moves as run_detect

    context = _build_detect_moves_context(session_id, sample_fps, clip_id)
    session = context["session"]
    calibration = context["calibration"]
    cap = session["cap"]
    fps = context["fps"]
    start_time = context["start_time"]
    sample_indices = context["sample_indices"]
    total_samples = len(sample_indices)

    fens: list[str | None] = []
    frame_indices: list[int] = []
    num_readable = 0
    config = VideoAnalysisConfig(reader_backend=reader_backend, scene_backend="none", device="cpu")
    sequence_reader: LockedOverlaySequenceReader | None = None
    locked_grid = None
    clip_data = context.get("clip_data")
    if reader_backend == "overlay" and clip_data is not None:
        locked_grid = _get_or_init_clip_grid(session, clip_data, calibration)

    lock = session["lock"]
    for completed_samples, idx in enumerate(sample_indices, start=1):
        with lock:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
        if not ret:
            fens.append(None)
            frame_indices.append(idx)
            if progress_callback is not None:
                progress_callback(completed_samples, total_samples, idx, num_readable)
            continue

        ox, oy, ow, oh = calibration.overlay
        overlay_img = frame[oy : oy + oh, ox : ox + ow]
        if reader_backend == "overlay":
            fen: str | None = None
            if sequence_reader is None:
                grid = locked_grid or find_board_grid_in_crop(overlay_img)
                if grid is not None:
                    sequence_reader = LockedOverlaySequenceReader(grid, device=config.device)
                    fen = sequence_reader.read(overlay_img).fen
            else:
                fen = sequence_reader.read(overlay_img).fen
            read_fen = fen
        else:
            read_fen = read_overlay_crop(overlay_img, config).fen

        fens.append(read_fen)
        frame_indices.append(idx)
        if read_fen is not None:
            num_readable += 1
        if progress_callback is not None:
            progress_callback(completed_samples, total_samples, idx, num_readable)

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
        "reader_backend": reader_backend,
        "segments": result_segments,
    }


def _build_detect_moves_context(
    session_id: str,
    sample_fps: float,
    clip_id: int | None,
) -> dict[str, Any]:
    """Resolve and validate the move-detection inputs."""
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

    fps = session["fps"]
    total_frames = session["total_frames"]
    start_frame = 0
    end_frame = total_frames
    start_time = 0.0
    if clip_data is not None:
        start_frame = int(clip_data["start_time"] * fps) if fps > 0 else 0
        if clip_data["end_time"] is not None:
            end_frame = int(clip_data["end_time"] * fps)
        start_time = clip_data["start_time"]

    frame_interval = max(1, int(fps / sample_fps))
    sample_indices = list(range(start_frame, end_frame, frame_interval))

    return {
        "session": session,
        "clip_data": clip_data,
        "calibration": calibration,
        "fps": fps,
        "start_time": start_time,
        "sample_indices": sample_indices,
    }


def start_detect_moves_job(
    session_id: str,
    sample_fps: float = 2.0,
    clip_id: int | None = None,
    reader_backend: str = "overlay",
) -> dict:
    """Start move detection in a background thread and return the job state."""
    context = _build_detect_moves_context(session_id, sample_fps, clip_id)
    total_samples = len(context["sample_indices"])

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _move_detection_jobs[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "running",
            "sample_fps": sample_fps,
            "clip_id": clip_id,
            "reader_backend": reader_backend,
            "error": None,
            "result": None,
            "total_samples": total_samples,
            "completed_samples": 0,
            "num_readable": 0,
            "current_frame_idx": None,
        }
        _persist_jobs(force=True)

    thread = threading.Thread(
        target=_run_detect_moves_job,
        args=(job_id, session_id, sample_fps, clip_id, reader_backend),
        daemon=True,
    )
    thread.start()
    return get_detect_moves_job(job_id, session_id)


def _run_detect_moves_job(
    job_id: str,
    session_id: str,
    sample_fps: float,
    clip_id: int | None,
    reader_backend: str,
) -> None:
    """Background job target for move detection."""
    try:
        result = detect_moves(
            session_id,
            sample_fps=sample_fps,
            clip_id=clip_id,
            reader_backend=reader_backend,
            progress_callback=lambda completed, total, frame_idx, num_readable: _update_detect_moves_job_progress(
                job_id,
                completed,
                total,
                frame_idx,
                num_readable,
            ),
        )
        with _jobs_lock:
            job = _move_detection_jobs.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = result
                job["completed_samples"] = job["total_samples"]
                job["num_readable"] = result["num_readable"]
            _persist_jobs(force=True)
    except Exception as exc:
        logger.exception("Move detection job %s failed", job_id)
        with _jobs_lock:
            job = _move_detection_jobs.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
            _persist_jobs(force=True)


def _update_detect_moves_job_progress(
    job_id: str,
    completed_samples: int,
    total_samples: int,
    frame_idx: int,
    num_readable: int,
) -> None:
    """Update progress counters for a running move-detection job."""
    with _jobs_lock:
        job = _move_detection_jobs.get(job_id)
        if job is None:
            return
        job["completed_samples"] = completed_samples
        job["total_samples"] = total_samples
        job["current_frame_idx"] = frame_idx
        job["num_readable"] = num_readable
        _persist_jobs()


def get_detect_moves_job(job_id: str, session_id: str | None = None) -> dict | None:
    """Return the current move-detection job status."""
    with _jobs_lock:
        job = _move_detection_jobs.get(job_id)
        if job is None:
            return None
        if session_id is not None and job["session_id"] != session_id:
            return None

        return {
            "job_id": job_id,
            "status": job["status"],
            "sample_fps": job["sample_fps"],
            "clip_id": job["clip_id"],
            "reader_backend": job["reader_backend"],
            "error": job["error"],
            "result": job["result"],
            "total_samples": job["total_samples"],
            "completed_samples": job["completed_samples"],
            "num_readable": job["num_readable"],
            "current_frame_idx": job["current_frame_idx"],
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


# ── Generate clips (background job) ───────────────────────────

_generate_jobs: dict[str, dict] = {}
_generate_jobs_lock = threading.Lock()


def start_generate_clips_job(
    session_id: str,
    clip_id: int | None = None,
) -> dict:
    """Start clip generation in a background thread."""
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

    job_id = uuid.uuid4().hex[:12]
    with _generate_jobs_lock:
        _generate_jobs[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "running",
            "clips": [],
            "error": None,
        }

    thread = threading.Thread(
        target=_run_generate_clips,
        args=(job_id, session, calibration, clip_data),
        daemon=True,
    )
    thread.start()
    return get_generate_clips_job(job_id)


def get_generate_clips_job(job_id: str) -> dict | None:
    with _generate_jobs_lock:
        job = _generate_jobs.get(job_id)
        if job is None:
            return None
        return dict(job)


def _run_generate_clips(
    job_id: str,
    session: dict,
    calibration: dict,
    clip_data: dict | None,
) -> None:
    from pipeline.overlay.overlay_clip_generator import OverlayClipGenerator

    try:
        video_path = session["video_path"]
        video_id = os.path.splitext(os.path.basename(video_path))[0]

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
            clip_info = {
                "filepath": r["filepath"],
                "num_frames": r["num_frames"],
                "num_moves": r["num_moves"],
                "game_index": r["game_index"],
                "pgn_moves": r.get("pgn_moves", ""),
            }
            clips.append(clip_info)
            with _generate_jobs_lock:
                job = _generate_jobs.get(job_id)
                if job:
                    job["clips"] = list(clips)

        with _generate_jobs_lock:
            job = _generate_jobs.get(job_id)
            if job:
                job["status"] = "done"
                job["clips"] = clips
    except Exception as e:
        with _generate_jobs_lock:
            job = _generate_jobs.get(job_id)
            if job:
                job["status"] = "failed"
                job["error"] = str(e)


def delete_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        session["cap"].release()
