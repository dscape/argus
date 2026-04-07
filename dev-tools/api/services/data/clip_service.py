"""Session-backed helpers for inspecting synthetic training clips."""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any

import chess
import cv2
import numpy as np
import torch

_sessions: dict[str, dict[str, Any]] = {}


def _get_vocab() -> tuple[Any | None, int, int]:
    try:
        from argus.chess.move_vocabulary import NO_MOVE_IDX, UNKNOWN_IDX, get_vocabulary

        return get_vocabulary(), NO_MOVE_IDX, UNKNOWN_IDX
    except ImportError:
        return None, 1968, 1969


def create_session(file_bytes: bytes, _filename: str) -> str:
    """Persist an uploaded clip to a temp file and cache its loaded tensors."""
    session_id = str(uuid.uuid4())[:8]
    tmp_dir = tempfile.mkdtemp(prefix="argus_clip_")
    tmp_path = os.path.join(tmp_dir, f"clip_{uuid.uuid4().hex}.pt")

    with open(tmp_path, "wb") as handle:
        handle.write(file_bytes)

    clip = torch.load(tmp_path, map_location="cpu", weights_only=True)
    _sessions[session_id] = {"clip": clip, "path": tmp_path}
    return session_id


def delete_session(session_id: str) -> None:
    session = _sessions.pop(session_id, None)
    if session is None:
        return

    try:
        os.unlink(session["path"])
        os.rmdir(os.path.dirname(session["path"]))
    except OSError:
        pass


def inspect(session_id: str) -> dict[str, Any]:
    """Return clip metadata, replay information, and tensor summaries."""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    clip = session["clip"]
    path = session["path"]
    vocab, no_move_idx, unknown_idx = _get_vocab()

    tensors: list[dict[str, Any]] = []
    for key in sorted(clip.keys()):
        value = clip[key]
        if isinstance(value, torch.Tensor):
            tensors.append(
                {
                    "name": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )

    num_frames = 0
    pixel_range = [0, 0]
    if "frames" in clip:
        frames = clip["frames"]
        num_frames = frames.shape[0]
        pixel_range = [int(frames.min().item()), int(frames.max().item())]

    moves: list[dict[str, Any]] = []
    total_moves = 0
    no_move_count = 0
    unknown_count = 0
    replay_valid = False
    replay_error: str | None = None
    final_fen: str | None = None

    if "move_targets" in clip:
        move_targets = clip["move_targets"]
        detect_targets = clip.get("detect_targets")

        for frame_index in range(move_targets.shape[0]):
            move_index = move_targets[frame_index].item()
            if move_index == no_move_idx:
                no_move_count += 1
                continue
            if move_index == unknown_idx:
                unknown_count += 1
                continue

            uci = vocab.index_to_uci(move_index) if vocab else f"idx_{move_index}"
            detect_value = None
            if detect_targets is not None:
                detect_value = float(detect_targets[frame_index].item())

            moves.append(
                {
                    "frame_index": frame_index,
                    "uci": uci,
                    "san": None,
                    "detect_value": detect_value,
                }
            )
            total_moves += 1

        if moves and vocab:
            fens = clip.get("fens")
            board = chess.Board(fens[0]) if fens and len(fens) > 0 else chess.Board()
            replay_valid = True

            for ply, move_data in enumerate(moves):
                try:
                    move = chess.Move.from_uci(move_data["uci"])
                except ValueError:
                    replay_valid = False
                    replay_error = f"Invalid UCI at ply {ply}: {move_data['uci']}"
                    break

                if move not in board.legal_moves:
                    replay_valid = False
                    replay_error = (
                        f"Illegal at ply {ply}: {move_data['uci']} "
                        f"(frame {move_data['frame_index']})"
                    )
                    break

                move_data["san"] = board.san(move)
                board.push(move)

            if replay_valid:
                final_fen = board.board_fen()

    avg_legal_moves = None
    if "legal_masks" in clip:
        legal_masks = clip["legal_masks"]
        avg_legal_moves = float(legal_masks.float().sum(dim=1).mean().item())

    return {
        "file_size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
        "tensors": tensors,
        "num_frames": num_frames,
        "pixel_range": pixel_range,
        "moves": moves,
        "total_moves": total_moves,
        "no_move_frames": no_move_count,
        "unknown_frames": unknown_count,
        "replay_valid": replay_valid,
        "replay_error": replay_error,
        "final_fen": final_fen,
        "avg_legal_moves": avg_legal_moves,
    }


def get_frame_png(session_id: str, frame_index: int) -> bytes:
    """Extract a clip frame and return it as PNG bytes."""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    frames = session["clip"].get("frames")
    if frames is None:
        raise ValueError("No frames in clip")
    if frame_index < 0 or frame_index >= frames.shape[0]:
        raise ValueError(f"Frame index {frame_index} out of range [0, {frames.shape[0]})")

    frame = frames[frame_index]
    if frame.dtype == torch.uint8:
        image = frame.permute(1, 2, 0).numpy()
    else:
        image = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode frame as PNG")

    return buffer.tobytes()
