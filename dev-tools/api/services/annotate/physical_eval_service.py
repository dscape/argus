"""Service layer for held-out physical-board validation annotations."""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np
import torch
from pipeline.overlay.overlay_move_detector import find_move_between_positions
from pipeline.overlay.replay import build_replay_board
from pipeline.physical import eval_dataset, splits
from pipeline.shared import SQUARE_CLASS_NAMES

from api.services.data import clip_service

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_REAL_CLIP_RE = re.compile(r"^clip_overlay_(?P<video_id>.+?)_clip(?P<clip_id>\d+)_\d+\.pt$")
_LABEL_INDEX_BY_NAME = {name: index for index, name in enumerate(SQUARE_CLASS_NAMES)}


def list_clip_files(
    clips_dir: str = "data/argus/train_real",
    *,
    limit: int = 200,
) -> dict[str, Any]:
    return _list_clip_files(
        eval_dataset,
        clips_dir,
        limit=limit,
        split_name=eval_dataset.DATASET_SPLIT,
    )



def get_annotation_summary() -> dict[str, Any]:
    return _get_annotation_summary(eval_dataset)



def get_frame_annotation(clip_path: str, frame_index: int) -> dict[str, Any] | None:
    return _get_frame_annotation(eval_dataset, clip_path, frame_index)



def get_move_corrections(session_id: str, clip_path: str) -> dict[str, Any]:
    return _get_move_corrections(eval_dataset, session_id, clip_path)



def delete_annotation(clip_path: str, frame_index: int) -> dict[str, Any] | None:
    return _delete_annotation(eval_dataset, clip_path, frame_index)


def rectify_frame(
    session_id: str,
    frame_index: int,
    corners: list[list[float]],
    *,
    output_size: int = eval_dataset.DEFAULT_BOARD_SIZE,
) -> dict[str, Any]:
    image_rgb = _get_clip_frame_rgb(session_id, frame_index)
    rectified_rgb = eval_dataset.rectify_board_image(
        image_rgb,
        corners,
        output_size=output_size,
    )
    return {
        "image_b64": _encode_rgb_png(rectified_rgb),
        "output_size": output_size,
    }


def save_annotation(
    session_id: str,
    clip_path: str,
    frame_index: int,
    corners: list[list[float]],
    labels: list[int | None],
    *,
    output_size: int = eval_dataset.DEFAULT_BOARD_SIZE,
) -> dict[str, Any]:
    return _save_annotation(
        eval_dataset,
        session_id,
        clip_path,
        frame_index,
        corners,
        labels,
        output_size=output_size,
    )


def _list_clip_files(
    dataset_module: Any,
    clips_dir: str,
    *,
    limit: int,
    split_name: str,
) -> dict[str, Any]:
    splits.ensure_annotation_layout_migrated()
    directory = _resolve_within_project(clips_dir)
    if not directory.exists():
        return {"clips_dir": str(directory.relative_to(_PROJECT_ROOT)), "clips": []}

    saved_frame_counts = dataset_module.get_saved_frame_counts_by_clip()
    clips: list[dict[str, Any]] = []
    for path in sorted(directory.glob("clip_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        match = _REAL_CLIP_RE.match(path.name)
        source_video_id = match.group("video_id") if match else None
        assigned_split = splits.get_source_video_split(source_video_id)
        if assigned_split is not None and assigned_split != split_name:
            continue

        stat = path.stat()
        relative_clip_path = str(path.relative_to(_PROJECT_ROOT))
        annotated_frame_count = saved_frame_counts.get(relative_clip_path, 0)
        num_frames = _get_clip_num_frames(path) if annotated_frame_count > 0 else None
        fully_annotated = bool(num_frames and annotated_frame_count >= num_frames)
        clips.append(
            {
                "filename": path.name,
                "clip_path": relative_clip_path,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "source_video_id": source_video_id,
                "clip_id": int(match.group("clip_id")) if match else None,
                "annotated_frame_count": annotated_frame_count,
                "num_frames": num_frames,
                "fully_annotated": fully_annotated,
                "assigned_split": assigned_split,
            }
        )
        if len(clips) >= limit:
            break

    return {
        "clips_dir": str(directory.relative_to(_PROJECT_ROOT)),
        "clips": clips,
    }



def _get_annotation_summary(dataset_module: Any) -> dict[str, Any]:
    return dataset_module.get_annotation_summary()



def _get_frame_annotation(
    dataset_module: Any,
    clip_path: str,
    frame_index: int,
) -> dict[str, Any] | None:
    resolved = _resolve_within_project(clip_path)
    relative_clip_path = str(resolved.relative_to(_PROJECT_ROOT))
    return dataset_module.load_board_annotation(relative_clip_path, frame_index)



def _get_move_corrections(
    dataset_module: Any,
    session_id: str,
    clip_path: str,
) -> dict[str, Any]:
    resolved = _resolve_within_project(clip_path)
    relative_clip_path = str(resolved.relative_to(_PROJECT_ROOT))
    clip_info = clip_service.inspect(session_id)
    annotations = dataset_module.list_board_annotations(relative_clip_path)
    return _build_move_corrections(clip_info, annotations)



def _delete_annotation(
    dataset_module: Any,
    clip_path: str,
    frame_index: int,
) -> dict[str, Any] | None:
    resolved = _resolve_within_project(clip_path)
    deleted = dataset_module.delete_board_annotation(
        str(resolved.relative_to(_PROJECT_ROOT)), frame_index
    )
    if not deleted:
        return None
    return dataset_module.get_annotation_summary()



def _save_annotation(
    dataset_module: Any,
    session_id: str,
    clip_path: str,
    frame_index: int,
    corners: list[list[float]],
    labels: list[int | None],
    *,
    output_size: int,
) -> dict[str, Any]:
    resolved_clip_path = _resolve_within_project(clip_path)
    image_rgb = _get_clip_frame_rgb(session_id, frame_index)
    source_video_id = _source_video_id_from_path(resolved_clip_path)
    annotation = dataset_module.save_board_annotation(
        image_rgb,
        clip_path=str(resolved_clip_path.relative_to(_PROJECT_ROOT)),
        frame_index=frame_index,
        source_video_id=source_video_id,
        corners=corners,
        labels=labels,
        output_size=output_size,
    )
    return {
        "annotation": annotation,
        "summary": dataset_module.get_annotation_summary(),
    }



def _build_move_corrections(
    clip_info: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    num_frames = int(clip_info.get("num_frames") or 0)
    original_moves = list(clip_info.get("moves") or [])
    original_frame_replay_fens = list(clip_info.get("frame_replay_fens") or [])
    frame_timestamps_seconds = list(clip_info.get("frame_timestamps_seconds") or [])

    original_move_by_frame: dict[int, dict[str, Any]] = {}
    for move in original_moves:
        try:
            frame_index = int(move.get("frame_index", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= frame_index < num_frames:
            original_move_by_frame[frame_index] = move

    original_display_fens_by_frame = [
        _get_original_display_fen(original_frame_replay_fens, original_move_by_frame, frame_index)
        for frame_index in range(num_frames)
    ]

    annotated_board_fens_by_frame: dict[int, str] = {}
    annotation_labels_by_frame: dict[int, list[int | None]] = {}
    manually_changed_frames: set[int] = set()
    for annotation in annotations:
        try:
            frame_index = int(annotation.get("frame_index", -1))
        except (TypeError, ValueError):
            continue
        if frame_index < 0 or frame_index >= num_frames:
            continue

        raw_labels = annotation.get("labels")
        if not isinstance(raw_labels, list) or len(raw_labels) != 64:
            continue

        annotation_labels_by_frame[frame_index] = list(raw_labels)
        original_board_fen = _extract_board_fen(original_display_fens_by_frame[frame_index])
        board_fen = _labels_to_board_fen(raw_labels, original_board_fen)
        if board_fen is None:
            continue

        annotated_board_fens_by_frame[frame_index] = board_fen
        if board_fen != original_board_fen:
            manually_changed_frames.add(frame_index)

    board = _starting_replay_board(clip_info, original_moves)
    if board is None:
        return {
            "frame_replay_fens": original_frame_replay_fens,
            "moves": original_moves,
            "total_moves": len(original_moves),
            "replay_valid": False,
            "replay_error": "Unable to build replay board for manual corrections",
            "manual_move_frames": [],
        }

    corrected_frame_replay_fens: list[str | None] = []
    corrected_moves: list[dict[str, Any]] = []
    replay_valid = True
    replay_errors: list[str] = []

    for frame_index in range(num_frames):
        corrected_frame_replay_fens.append(board.fen())

        target_full_fen = original_display_fens_by_frame[frame_index]
        target_board_fen = annotated_board_fens_by_frame.get(frame_index)
        if target_board_fen is None:
            target_board_fen = _extract_board_fen(target_full_fen)

        if target_board_fen is None or board.board_fen() == target_board_fen:
            continue

        move = _find_move_with_turn_fallback(board, target_board_fen)
        if move is None:
            move = _find_move_matching_labels_with_turn_fallback(
                board,
                annotation_labels_by_frame.get(frame_index),
            )
        if move is None:
            replay_valid = False
            replay_errors.append(
                f"Could not infer a legal move for frame {frame_index + 1}"
            )
            board = _resync_board(board, target_board_fen, target_full_fen)
            continue

        original_move = original_move_by_frame.get(frame_index)
        is_manual = (
            frame_index in manually_changed_frames
            or original_move is None
            or original_move.get("uci") != move.uci()
        )

        fen_before = board.fen()
        side_to_move = "white" if board.turn == chess.WHITE else "black"
        san = board.san(move)
        board.push(move)

        corrected_moves.append(
            {
                "frame_index": frame_index,
                "uci": move.uci(),
                "san": san,
                "detect_value": original_move.get("detect_value") if original_move else None,
                "timestamp_seconds": _move_timestamp(
                    original_move,
                    frame_timestamps_seconds,
                    frame_index,
                ),
                "estimated_otb_frame_index": (
                    original_move.get("estimated_otb_frame_index") if original_move else None
                ),
                "estimated_otb_timestamp_seconds": (
                    original_move.get("estimated_otb_timestamp_seconds") if original_move else None
                ),
                "side_to_move": side_to_move,
                "fen_before": fen_before,
                "fen_after": board.fen(),
                "is_manual": is_manual,
            }
        )

    manual_move_frames = [
        int(move["frame_index"])
        for move in corrected_moves
        if bool(move.get("is_manual"))
    ]

    return {
        "frame_replay_fens": corrected_frame_replay_fens,
        "moves": corrected_moves,
        "total_moves": len(corrected_moves),
        "replay_valid": replay_valid,
        "replay_error": "; ".join(replay_errors[:3]) if replay_errors else None,
        "manual_move_frames": manual_move_frames,
    }


def _resolve_within_project(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (_PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.is_relative_to(_PROJECT_ROOT):
        raise ValueError(f"Path is outside the project root: {path}")
    return candidate


def _get_clip_num_frames(path: Path) -> int | None:
    try:
        clip = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None

    frames = clip.get("frames") if isinstance(clip, dict) else None
    if not isinstance(frames, torch.Tensor) or frames.ndim < 1:
        return None
    return int(frames.shape[0])


def _get_clip_frame_rgb(session_id: str, frame_index: int) -> np.ndarray:
    session = clip_service.get_session(session_id)
    if session is None:
        raise ValueError("Clip session not found")

    clip = session["clip"]
    frames = clip.get("frames")
    if frames is None:
        raise ValueError("No frames in clip")
    if frame_index < 0 or frame_index >= frames.shape[0]:
        raise ValueError(f"Frame index {frame_index} out of range [0, {frames.shape[0]})")

    frame = frames[frame_index]
    if frame.dtype == torch.uint8:
        image_rgb = frame.permute(1, 2, 0).numpy()
    else:
        image_rgb = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return image_rgb


def _encode_rgb_png(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encoded, buffer = cv2.imencode(".png", image_bgr)
    if not encoded:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _source_video_id_from_path(path: Path) -> str | None:
    match = _REAL_CLIP_RE.match(path.name)
    return match.group("video_id") if match else None


def _get_original_display_fen(
    frame_replay_fens: list[str | None],
    moves_by_frame: dict[int, dict[str, Any]],
    frame_index: int,
) -> str | None:
    move = moves_by_frame.get(frame_index)
    if move is not None and isinstance(move.get("fen_after"), str):
        return move["fen_after"]
    if frame_index < len(frame_replay_fens) and isinstance(frame_replay_fens[frame_index], str):
        return frame_replay_fens[frame_index]
    return None


def _starting_replay_board(
    clip_info: dict[str, Any],
    original_moves: list[dict[str, Any]],
) -> chess.Board | None:
    frame_replay_fens = clip_info.get("frame_replay_fens") or []
    for fen in frame_replay_fens:
        if not isinstance(fen, str):
            continue
        try:
            return chess.Board(fen)
        except ValueError:
            continue

    metadata = clip_info.get("metadata") if isinstance(clip_info.get("metadata"), dict) else {}
    initial_board_fen = metadata.get("initial_board_fen")
    if not isinstance(initial_board_fen, str):
        return None

    first_move_uci = next(
        (move.get("uci") for move in original_moves if isinstance(move.get("uci"), str)),
        None,
    )
    return build_replay_board(initial_board_fen, first_move_uci)


def _extract_board_fen(fen: str | None) -> str | None:
    if not isinstance(fen, str):
        return None
    return fen.split(" ", 1)[0]


def _labels_to_board_fen(labels: Any, base_board_fen: str | None = None) -> str | None:
    if not isinstance(labels, list) or len(labels) != 64:
        return None

    base_labels = _board_fen_to_labels(base_board_fen)
    resolved_labels: list[int] = []
    for index, raw_label in enumerate(labels):
        if raw_label is None:
            if base_labels is None:
                return None
            resolved_labels.append(base_labels[index])
            continue
        if not isinstance(raw_label, int) or raw_label < 0 or raw_label >= len(SQUARE_CLASS_NAMES):
            return None
        resolved_labels.append(raw_label)

    squares: list[str] = []
    for rank in range(8):
        empty_run = 0
        rank_tokens: list[str] = []
        for file in range(8):
            label = resolved_labels[rank * 8 + file]
            if label == 0:
                empty_run += 1
                continue
            if empty_run:
                rank_tokens.append(str(empty_run))
                empty_run = 0
            rank_tokens.append(SQUARE_CLASS_NAMES[label])
        if empty_run:
            rank_tokens.append(str(empty_run))
        squares.append("".join(rank_tokens) or "8")

    return "/".join(squares)


def _board_fen_to_labels(board_fen: str | None) -> list[int] | None:
    if not isinstance(board_fen, str):
        return None

    labels: list[int] = []
    label_index_by_name = {name: index for index, name in enumerate(SQUARE_CLASS_NAMES)}
    for rank in board_fen.split("/"):
        for token in rank:
            if token.isdigit():
                labels.extend([0] * int(token))
                continue
            label = label_index_by_name.get(token)
            if label is None:
                return None
            labels.append(label)

    if len(labels) != 64:
        return None
    return labels


def _find_move_with_turn_fallback(
    board: chess.Board,
    target_board_fen: str,
) -> chess.Move | None:
    move = find_move_between_positions(board, target_board_fen)
    if move is not None:
        return move

    fallback_board = board.copy(stack=False)
    fallback_board.turn = not board.turn
    move = find_move_between_positions(fallback_board, target_board_fen)
    if move is None:
        return None

    board.turn = fallback_board.turn
    return move


def _find_move_matching_labels_with_turn_fallback(
    board: chess.Board,
    labels: list[int | None] | None,
) -> chess.Move | None:
    move = _find_move_matching_labels(board, labels)
    if move is not None:
        return move

    fallback_board = board.copy(stack=False)
    fallback_board.turn = not board.turn
    move = _find_move_matching_labels(fallback_board, labels)
    if move is None:
        return None

    board.turn = fallback_board.turn
    return move


def _find_move_matching_labels(
    board: chess.Board,
    labels: list[int | None] | None,
) -> chess.Move | None:
    if labels is None or len(labels) != 64:
        return None

    matching_moves: list[chess.Move] = []
    for move in board.legal_moves:
        test_board = board.copy(stack=False)
        test_board.push(move)
        if _board_matches_labels(test_board, labels):
            matching_moves.append(move)
            if len(matching_moves) > 1:
                return None

    return matching_moves[0] if matching_moves else None


def _board_matches_labels(board: chess.Board, labels: list[int | None]) -> bool:
    for square_index, raw_label in enumerate(labels):
        if raw_label is None:
            continue
        if not isinstance(raw_label, int) or raw_label < 0 or raw_label >= len(SQUARE_CLASS_NAMES):
            return False
        if _board_label_at(board, square_index) != raw_label:
            return False
    return True


def _board_label_at(board: chess.Board, square_index: int) -> int:
    file_index = square_index % 8
    rank_index = 7 - (square_index // 8)
    piece = board.piece_at(chess.square(file_index, rank_index))
    if piece is None:
        return 0
    return _LABEL_INDEX_BY_NAME[piece.symbol()]


def _move_timestamp(
    original_move: dict[str, Any] | None,
    frame_timestamps_seconds: list[float],
    frame_index: int,
) -> float | None:
    if original_move is not None and isinstance(
        original_move.get("timestamp_seconds"),
        (int, float),
    ):
        return float(original_move["timestamp_seconds"])
    if frame_index < len(frame_timestamps_seconds):
        return float(frame_timestamps_seconds[frame_index])
    return None


def _resync_board(
    board: chess.Board,
    target_board_fen: str,
    target_full_fen: str | None,
) -> chess.Board:
    if isinstance(target_full_fen, str) and _extract_board_fen(target_full_fen) == target_board_fen:
        try:
            return chess.Board(target_full_fen)
        except ValueError:
            pass

    resynced = chess.Board()
    resynced.set_board_fen(target_board_fen)
    resynced.turn = not board.turn
    resynced.halfmove_clock = 0
    resynced.fullmove_number = board.fullmove_number + (1 if board.turn == chess.BLACK else 0)
    return resynced
