"""Service layer for clip inspection, wrapping diagnostics.inspect_clip logic."""

import os
import tempfile
import uuid
from typing import Any

import chess
import cv2
import numpy as np
import torch

# Session storage: session_id -> {"clip": dict, "path": str}
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


def create_session(file_bytes: bytes, filename: str) -> str:
    """Save clip to temp file, load it, and return a session_id."""
    session_id = str(uuid.uuid4())[:8]
    tmp_dir = tempfile.mkdtemp(prefix="argus_clip_")
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    clip = torch.load(tmp_path, map_location="cpu", weights_only=False)
    _sessions[session_id] = {"clip": clip, "path": tmp_path}
    return session_id


def get_session(session_id: str) -> dict[str, Any] | None:
    return _sessions.get(session_id)


def delete_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if session:
        try:
            os.unlink(session["path"])
            os.rmdir(os.path.dirname(session["path"]))
        except OSError:
            pass


def inspect(session_id: str) -> dict:
    """Inspect a loaded clip and return structured data."""
    session = _sessions.get(session_id)
    if session is None:
        raise ValueError("Session not found")

    clip = session["clip"]
    path = session["path"]
    vocab, NO_MOVE_IDX, UNKNOWN_IDX = _get_vocab()

    # Tensor info
    tensors = []
    for key in sorted(clip.keys()):
        val = clip[key]
        if isinstance(val, torch.Tensor):
            tensors.append({
                "name": key,
                "shape": list(val.shape),
                "dtype": str(val.dtype),
            })

    # Frame info
    num_frames = 0
    pixel_range = [0, 0]
    if "frames" in clip:
        frames = clip["frames"]
        num_frames = frames.shape[0]
        pixel_range = [int(frames.min().item()), int(frames.max().item())]

    # Move analysis
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
        T = move_targets.shape[0]

        for t in range(T):
            idx = move_targets[t].item()
            if idx == NO_MOVE_IDX:
                no_move_count += 1
            elif idx == UNKNOWN_IDX:
                unknown_count += 1
            else:
                uci = vocab.index_to_uci(idx) if vocab else f"idx_{idx}"
                detect = None
                if detect_targets is not None:
                    detect = float(detect_targets[t].item())
                moves.append({
                    "frame_index": t,
                    "uci": uci,
                    "san": None,
                    "detect_value": detect,
                })
                total_moves += 1

        # Replay validation
        if moves and vocab:
            board = chess.Board()
            replay_valid = True
            for i, m in enumerate(moves):
                try:
                    move = chess.Move.from_uci(m["uci"])
                    if move not in board.legal_moves:
                        replay_valid = False
                        replay_error = f"Illegal at ply {i}: {m['uci']} (frame {m['frame_index']})"
                        break
                    m["san"] = board.san(move)
                    board.push(move)
                except ValueError:
                    replay_valid = False
                    replay_error = f"Invalid UCI at ply {i}: {m['uci']}"
                    break

            if replay_valid:
                final_fen = board.board_fen()

    # Legal masks
    avg_legal = None
    if "legal_masks" in clip:
        legal_masks = clip["legal_masks"]
        avg_legal = float(legal_masks.float().sum(dim=1).mean().item())

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
        "avg_legal_moves": avg_legal,
    }


def get_frame_png(session_id: str, frame_index: int) -> bytes:
    """Extract a single frame as PNG bytes."""
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
        img = frame.permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
    else:
        img = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    # RGB -> BGR for cv2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()
