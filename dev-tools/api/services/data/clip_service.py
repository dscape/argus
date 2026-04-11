"""Service layer for clip inspection, wrapping diagnostics.inspect_clip logic."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np
import torch

from pipeline.overlay.replay import build_replay_board

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_ANNOTATIONS_ROOT = _PROJECT_ROOT / "data" / "clip_annotations"
_REAL_DB_CLIP_RE = re.compile(r"^clip_overlay_(?P<video_id>.+?)_clip(?P<clip_id>\d+)_\d+\.pt$")

# Session storage: session_id -> {"clip": dict, "path": str, ...}
_sessions: dict[str, dict[str, Any]] = {}


def _get_vocab():
    try:
        from argus.chess.move_vocabulary import (
            NO_MOVE_IDX,
            UNKNOWN_IDX,
            get_vocabulary,
        )

        return get_vocabulary(), NO_MOVE_IDX, UNKNOWN_IDX
    except ImportError:
        return None, 1968, 1969


def _tensor_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return []


def _tensor_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return []


def _safe_filename(filename: str) -> str:
    return Path(filename).name


def _annotation_path(filename: str) -> Path:
    safe_name = _safe_filename(filename)
    stem = Path(safe_name).stem or safe_name
    return _ANNOTATIONS_ROOT / f"{stem}.txt"


def _parse_real_clip_reference(filename: str | None) -> tuple[str, int] | None:
    if not filename:
        return None
    match = _REAL_DB_CLIP_RE.match(filename)
    if not match:
        return None
    return match.group("video_id"), int(match.group("clip_id"))


def _load_db_clip_overlay_row(clip_id: int) -> dict[str, Any] | None:
    try:
        from pipeline.db.connection import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, video_id, overlay_bbox, ref_resolution
                    FROM video_clips
                    WHERE id = %s
                    """,
                    (clip_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {
                    "id": int(row[0]),
                    "video_id": row[1],
                    "overlay_bbox": tuple(int(v) for v in row[2]),
                    "ref_resolution": tuple(int(v) for v in row[3]),
                }
    except Exception:
        return None


def _get_video_path(video_id: str) -> str | None:
    try:
        from pipeline.download.video_downloader import get_video_path

        return get_video_path(video_id)
    except Exception:
        return None


def _get_source_video_path_for_clip(clip: dict[str, Any]) -> str | None:
    source_video_id = clip.get("source_video_id")
    if not isinstance(source_video_id, str) or not source_video_id:
        return None
    return _get_video_path(source_video_id)


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    ref_resolution: tuple[int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    ref_width, ref_height = ref_resolution
    scale_x = width / ref_width if ref_width > 0 else 1.0
    scale_y = height / ref_height if ref_height > 0 else 1.0
    x, y, w, h = bbox
    scaled = (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        int(round(w * scale_x)),
        int(round(h * scale_y)),
    )
    sx, sy, sw, sh = scaled
    sx = max(0, min(sx, width - 1))
    sy = max(0, min(sy, height - 1))
    sw = max(1, min(sw, width - sx))
    sh = max(1, min(sh, height - sy))
    return sx, sy, sw, sh


def _get_overlay_preview_context(session: dict[str, Any]) -> dict[str, Any] | None:
    clip = session["clip"]
    clip_reference = _parse_real_clip_reference(session.get("filename"))
    if clip_reference is None:
        return None

    video_id, clip_id = clip_reference
    source_video_id = clip.get("source_video_id")
    if isinstance(source_video_id, str) and source_video_id != video_id:
        return None

    frame_indices = clip.get("frame_indices")
    if not isinstance(frame_indices, torch.Tensor) or frame_indices.ndim != 1:
        return None

    clip_row = _load_db_clip_overlay_row(clip_id)
    if clip_row is None or clip_row["video_id"] != video_id:
        return None

    video_path = _get_video_path(video_id)
    if video_path is None:
        return None

    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "video_path": video_path,
        "frame_indices": frame_indices,
        "overlay_bbox": clip_row["overlay_bbox"],
        "ref_resolution": clip_row["ref_resolution"],
    }


def create_session(
    file_bytes: bytes,
    filename: str,
    *,
    source_filepath: str | None = None,
) -> str:
    """Save clip to temp file, load it, and return a session_id."""
    session_id = str(uuid.uuid4())[:8]
    tmp_dir = tempfile.mkdtemp(prefix="argus_clip_")
    safe_name = f"clip_{uuid.uuid4().hex}.pt"
    tmp_path = os.path.join(tmp_dir, safe_name)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    clip = torch.load(tmp_path, map_location="cpu", weights_only=True)
    _sessions[session_id] = {
        "clip": clip,
        "path": tmp_path,
        "filename": filename,
        "source_filepath": source_filepath,
    }
    return session_id


def get_session(session_id: str) -> dict[str, Any] | None:
    return _sessions.get(session_id)


def delete_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        temp_dir = Path(session["path"]).parent
        try:
            for child in temp_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
            temp_dir.rmdir()
        except OSError:
            pass


def _ensure_review_video_path(session: dict[str, Any]) -> str:
    cached_path = session.get("review_video_path")
    if isinstance(cached_path, str) and os.path.exists(cached_path):
        return cached_path

    source_video_path = _get_source_video_path_for_clip(session["clip"])
    if source_video_path is None:
        raise ValueError("Source video is unavailable for this clip")

    review_video_path = Path(session["path"]).with_name("source_review.mp4")
    if review_video_path.exists():
        session["review_video_path"] = str(review_video_path)
        return str(review_video_path)

    source_suffix = Path(source_video_path).suffix.lower()
    if source_suffix == ".mp4":
        command = [
            "ffmpeg",
            "-y",
            "-i",
            source_video_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(review_video_path),
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            source_video_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "27",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(review_video_path),
        ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Failed to prepare review video: {exc.stderr.strip()}") from exc

    session["review_video_path"] = str(review_video_path)
    return str(review_video_path)


def get_source_video_path(session_id: str) -> str:
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")
    return _ensure_review_video_path(session)


def get_annotation(filename: str) -> dict[str, Any]:
    path = _annotation_path(filename)
    exists = path.exists()
    return {
        "filename": _safe_filename(filename),
        "annotation_path": str(path),
        "content": path.read_text() if exists else "",
        "exists": exists,
    }


def save_annotation(filename: str, content: str) -> dict[str, Any]:
    path = _annotation_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return {
        "filename": _safe_filename(filename),
        "annotation_path": str(path),
        "content": content,
        "exists": True,
    }


def inspect(session_id: str) -> dict[str, Any]:
    """Inspect a loaded clip and return structured data."""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    clip = session["clip"]
    path = session["path"]
    vocab, no_move_idx, unknown_idx = _get_vocab()

    tensors = []
    for key in sorted(clip.keys()):
        val = clip[key]
        if isinstance(val, torch.Tensor):
            tensors.append(
                {
                    "name": key,
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                }
            )

    num_frames = 0
    pixel_range = [0, 0]
    if "frames" in clip:
        frames = clip["frames"]
        num_frames = frames.shape[0]
        pixel_range = [int(frames.min().item()), int(frames.max().item())]

    frame_indices = _tensor_int_list(clip.get("frame_indices"))
    if len(frame_indices) != num_frames:
        frame_indices = list(range(num_frames))

    frame_timestamps_seconds = _tensor_float_list(clip.get("frame_timestamps_seconds"))
    if len(frame_timestamps_seconds) != num_frames:
        frame_timestamps_seconds = []

    move_timestamps_seconds = _tensor_float_list(clip.get("move_timestamps_seconds"))

    moves = []
    total_moves = 0
    no_move_count = 0
    unknown_count = 0
    replay_valid = False
    replay_error = None
    final_fen = None

    if "move_targets" in clip:
        move_targets = clip["move_targets"]
        detect_targets = clip.get("detect_targets")
        total_frames = move_targets.shape[0]
        move_event_index = 0

        for frame_index in range(total_frames):
            idx = move_targets[frame_index].item()
            if idx == no_move_idx:
                no_move_count += 1
            elif idx == unknown_idx:
                unknown_count += 1
            else:
                uci = vocab.index_to_uci(idx) if vocab else f"idx_{idx}"
                detect = None
                if detect_targets is not None:
                    detect = float(detect_targets[frame_index].item())
                timestamp_seconds = None
                if move_event_index < len(move_timestamps_seconds):
                    timestamp_seconds = move_timestamps_seconds[move_event_index]
                moves.append(
                    {
                        "frame_index": frame_index,
                        "uci": uci,
                        "san": None,
                        "detect_value": detect,
                        "timestamp_seconds": timestamp_seconds,
                    }
                )
                move_event_index += 1
                total_moves += 1

        if moves and vocab:
            initial_board_fen = clip.get("initial_board_fen")
            if isinstance(initial_board_fen, str):
                board = build_replay_board(initial_board_fen, moves[0]["uci"])
            else:
                fens = clip.get("fens")
                if fens and len(fens) > 0:
                    board = chess.Board(fens[0])
                else:
                    board = chess.Board()
            replay_valid = True
            for i, move_data in enumerate(moves):
                try:
                    move = chess.Move.from_uci(move_data["uci"])
                    if move not in board.legal_moves:
                        replay_valid = False
                        replay_error = (
                            f"Illegal at ply {i}: {move_data['uci']} "
                            f"(frame {move_data['frame_index']})"
                        )
                        break
                    move_data["san"] = board.san(move)
                    board.push(move)
                except ValueError:
                    replay_valid = False
                    replay_error = f"Invalid UCI at ply {i}: {move_data['uci']}"
                    break

            if replay_valid:
                final_fen = board.board_fen()

    avg_legal = None
    if "legal_masks" in clip:
        legal_masks = clip["legal_masks"]
        avg_legal = float(legal_masks.float().sum(dim=1).mean().item())

    metadata: dict[str, Any] = {}
    for key in [
        "initial_board_fen",
        "initial_side_to_move",
        "pgn_moves",
        "source_video_id",
        "source_channel_handle",
    ]:
        value = clip.get(key)
        if isinstance(value, (str, bool, int, float)) or value is None:
            metadata[key] = value
    for key in [
        "segment_start_time_seconds",
        "segment_end_time_seconds",
        "sampled_video_fps",
        "num_moves",
    ]:
        value = clip.get(key)
        if isinstance(value, (bool, int, float)) or value is None:
            metadata[key] = value

    clip_reference = _parse_real_clip_reference(session.get("filename"))
    if clip_reference is not None:
        _, clip_id = clip_reference
        metadata["source_db_clip_id"] = clip_id

    return {
        "file_size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
        "tensors": tensors,
        "num_frames": num_frames,
        "frame_indices": frame_indices,
        "frame_timestamps_seconds": frame_timestamps_seconds,
        "pixel_range": pixel_range,
        "moves": moves,
        "total_moves": total_moves,
        "no_move_frames": no_move_count,
        "unknown_frames": unknown_count,
        "replay_valid": replay_valid,
        "replay_error": replay_error,
        "final_fen": final_fen,
        "avg_legal_moves": avg_legal,
        "metadata": metadata,
    }


def get_frame_png(session_id: str, frame_index: int) -> bytes:
    """Extract a single stored clip frame as PNG bytes."""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    clip = session["clip"]
    frames = clip.get("frames")
    if frames is None:
        raise ValueError("No frames in clip")

    if frame_index < 0 or frame_index >= frames.shape[0]:
        raise ValueError(f"Frame index {frame_index} out of range [0, {frames.shape[0]})")

    frame = frames[frame_index]
    if frame.dtype == torch.uint8:
        img = frame.permute(1, 2, 0).numpy()
    else:
        img = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode clip frame")
    return buffer.tobytes()


def get_overlay_frame_png(session_id: str, frame_index: int) -> bytes:
    """Extract the corresponding overlay crop from the source video as PNG bytes."""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    context = _get_overlay_preview_context(session)
    if context is None:
        raise ValueError("Overlay preview is unavailable for this clip")

    source_frame_indices: torch.Tensor = context["frame_indices"]
    if frame_index < 0 or frame_index >= source_frame_indices.shape[0]:
        raise ValueError(
            f"Frame index {frame_index} out of range [0, {source_frame_indices.shape[0]})"
        )

    cap = cv2.VideoCapture(context["video_path"])
    if not cap.isOpened():
        raise ValueError("Failed to open source video")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bbox = _scale_bbox(
            context["overlay_bbox"],
            context["ref_resolution"],
            width,
            height,
        )

        source_frame_index = int(source_frame_indices[frame_index].item())
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Failed to read source video frame")

        x, y, w, h = bbox
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            raise ValueError("Overlay crop was empty")

        encoded, buffer = cv2.imencode(".png", crop)
        if not encoded:
            raise ValueError("Failed to encode overlay frame")
        return buffer.tobytes()
    finally:
        cap.release()
