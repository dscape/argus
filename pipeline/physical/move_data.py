"""Physical move-model datasets built from real clip metadata and held-out annotations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import cv2
import torch

from argus.chess.constraint_mask import get_legal_mask
from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from pipeline.overlay.overlay_move_detector import find_move_between_positions
from pipeline.physical import splits
from pipeline.physical.annotation_dataset import rectify_board_image
from pipeline.physical.oblique_square_context import (
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.real_board_data import (
    _build_replay_board,
    _frame_tensor_to_rgb,
    _int_list,
    _string_list,
    load_real_board_rows,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_DEFAULT_EVAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"
_DEFAULT_IMAGE_SIZE = 224


@dataclass(frozen=True)
class PhysicalMoveWindowMetadata:
    clip_path: str
    source_video_id: str | None
    start_index: int
    end_index: int
    contains_move: bool


@dataclass(frozen=True)
class PhysicalEvalMoveSequence:
    clip_path: str
    source_video_id: str | None
    initial_board_fen: str
    frames: torch.Tensor
    frame_indices: tuple[int, ...]
    labels: tuple[tuple[int, ...], ...]
    inferred_move_targets: tuple[int, ...]
    inferred_detect_targets: tuple[float, ...]


def build_real_move_window_clips(
    *,
    clips_dir: str | Path = _DEFAULT_CLIPS_DIR,
    eval_root: str | Path = _DEFAULT_EVAL_ROOT,
    image_size: int = _DEFAULT_IMAGE_SIZE,
    clip_length: int = 16,
    negative_window_stride: int = 8,
    max_negative_windows_per_clip: int = 4,
    selection_source_video_ids: set[str] | None = None,
    exclude_selection_source_video_ids: bool = False,
) -> tuple[list[dict[str, Any]], list[PhysicalMoveWindowMetadata]]:
    """Build fixed-length real physical move windows from replay metadata."""
    splits.ensure_annotation_layout_migrated()
    if clip_length <= 0:
        raise ValueError(f"clip_length must be > 0, got {clip_length}")
    if negative_window_stride <= 0:
        raise ValueError(f"negative_window_stride must be > 0, got {negative_window_stride}")

    rows = load_real_board_rows(
        clips_dir=clips_dir,
        eval_root=eval_root,
        frame_stride=1,
        max_frames=None,
    )
    rows_by_clip: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        rows_by_clip[row.clip_path].append(row)

    clips: list[dict[str, Any]] = []
    metadata: list[PhysicalMoveWindowMetadata] = []
    for clip_path, clip_rows in rows_by_clip.items():
        clip_rows.sort(key=lambda row: row.frame_index)
        source_video_id = clip_rows[0].source_video_id
        if selection_source_video_ids:
            in_selection = source_video_id in selection_source_video_ids
            if exclude_selection_source_video_ids and in_selection:
                continue
            if not exclude_selection_source_video_ids and not in_selection:
                continue

        clip_payload = torch.load(_PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
        if not isinstance(clip_payload, dict):
            raise ValueError(f"Invalid clip payload: {clip_path}")

        clip_data = _build_real_clip_sample(
            clip_payload,
            clip_path=clip_path,
            clip_rows=clip_rows,
            image_size=image_size,
        )
        if clip_data is None:
            continue

        clip_windows = _slice_move_windows(
            clip_data,
            clip_length=clip_length,
            negative_window_stride=negative_window_stride,
            max_negative_windows=max_negative_windows_per_clip,
        )
        for window_payload, start_index, end_index, contains_move in clip_windows:
            clips.append(window_payload)
            metadata.append(
                PhysicalMoveWindowMetadata(
                    clip_path=clip_path,
                    source_video_id=source_video_id,
                    start_index=start_index,
                    end_index=end_index,
                    contains_move=contains_move,
                )
            )
    return clips, metadata


def load_eval_move_sequences(
    *,
    annotation_root: str | Path = _DEFAULT_EVAL_ROOT,
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> list[PhysicalEvalMoveSequence]:
    """Load held-out annotated sequences for move-model evaluation."""
    rows = load_annotated_oblique_rows(annotation_root)
    rows_by_clip: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        rows_by_clip[row.clip_path].append(row)

    clip_cache: dict[Path, dict[str, object]] = {}
    sequences: list[PhysicalEvalMoveSequence] = []
    for clip_path, clip_rows in rows_by_clip.items():
        clip_rows.sort(key=lambda row: row.frame_index)
        initial_board_fen = _initial_board_fen_for_clip(clip_path)
        inference = _infer_annotated_moves(clip_rows, initial_board_fen)

        frames = []
        labels = []
        frame_indices = []
        for row in clip_rows:
            image_bgr = _load_clip_frame_bgr(row, clip_cache=clip_cache)
            rectified = rectify_board_image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                [list(point) for point in row.corners],
                output_size=image_size,
            )
            frames.append(torch.from_numpy(rectified).permute(2, 0, 1).float() / 255.0)
            labels.append(tuple(int(value) for value in row.labels))
            frame_indices.append(int(row.frame_index))

        sequences.append(
            PhysicalEvalMoveSequence(
                clip_path=clip_path,
                source_video_id=clip_rows[0].source_video_id,
                initial_board_fen=initial_board_fen,
                frames=torch.stack(frames, dim=0),
                frame_indices=tuple(frame_indices),
                labels=tuple(labels),
                inferred_move_targets=tuple(inference[0]),
                inferred_detect_targets=tuple(inference[1]),
            )
        )
    sequences.sort(key=lambda sequence: sequence.clip_path)
    return sequences


def _build_real_clip_sample(
    clip: dict[str, Any],
    *,
    clip_path: str,
    clip_rows: list[Any],
    image_size: int,
) -> dict[str, Any] | None:
    frames = clip.get("frames")
    initial_board_fen = clip.get("initial_board_fen")
    if not isinstance(frames, torch.Tensor) or not isinstance(initial_board_fen, str):
        return None

    corners = clip_rows[0].corners
    replay_targets = _replay_targets_for_clip(clip)
    if replay_targets is None:
        return None
    move_targets, detect_targets, legal_masks, board_fens = replay_targets
    if int(frames.shape[0]) != len(move_targets):
        return None

    rectified_frames: list[torch.Tensor] = []
    for frame_index in range(int(frames.shape[0])):
        image_rgb = _frame_tensor_to_rgb(frames[frame_index])
        rectified = rectify_board_image(
            image_rgb,
            [list(point) for point in corners],
            output_size=image_size,
        )
        rectified_frames.append(torch.from_numpy(rectified).permute(2, 0, 1).float() / 255.0)

    return {
        "clip_path": clip_path,
        "frames": torch.stack(rectified_frames, dim=0),
        "move_targets": move_targets,
        "detect_targets": detect_targets,
        "legal_masks": legal_masks,
        "move_mask": detect_targets > 0.5,
        "fens": board_fens,
    }


def _slice_move_windows(
    clip_data: dict[str, Any],
    *,
    clip_length: int,
    negative_window_stride: int,
    max_negative_windows: int,
) -> list[tuple[dict[str, Any], int, int, bool]]:
    frames = clip_data["frames"]
    move_targets = clip_data["move_targets"]
    detect_targets = clip_data["detect_targets"]
    legal_masks = clip_data["legal_masks"]
    board_fens = clip_data.get("fens")
    total_frames = int(frames.shape[0])
    if total_frames == 0:
        return []

    window_starts: dict[int, bool] = {}
    move_indices = torch.nonzero(detect_targets > 0.5, as_tuple=False).reshape(-1).tolist()
    for move_index in move_indices:
        window_start = _window_start_for_index(
            move_index, total_frames=total_frames, clip_length=clip_length
        )
        window_starts[window_start] = True

    negative_count = 0
    for window_start in range(0, max(total_frames - clip_length + 1, 1), negative_window_stride):
        window_end = min(window_start + clip_length, total_frames)
        if float(detect_targets[window_start:window_end].sum().item()) > 0.0:
            continue
        if window_start in window_starts:
            continue
        if negative_count >= max_negative_windows:
            break
        window_starts[window_start] = False
        negative_count += 1

    windows: list[tuple[dict[str, Any], int, int, bool]] = []
    for window_start, contains_move in sorted(window_starts.items()):
        window_end = min(window_start + clip_length, total_frames)
        payload: dict[str, Any] = {
            "frames": frames[window_start:window_end].clone(),
            "move_targets": move_targets[window_start:window_end].clone(),
            "detect_targets": detect_targets[window_start:window_end].clone(),
            "legal_masks": legal_masks[window_start:window_end].clone(),
            "move_mask": (detect_targets[window_start:window_end] > 0.5).clone(),
        }
        if isinstance(board_fens, list):
            payload["fens"] = list(board_fens[window_start:window_end])
        windows.append((payload, window_start, window_end, contains_move))
    return windows


def _window_start_for_index(index: int, *, total_frames: int, clip_length: int) -> int:
    if total_frames <= clip_length:
        return 0
    centered = index - clip_length // 2
    return max(0, min(centered, total_frames - clip_length))


def _replay_targets_for_clip(
    clip: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]] | None:
    frames = clip.get("frames")
    initial_board_fen = clip.get("initial_board_fen")
    if not isinstance(frames, torch.Tensor) or not isinstance(initial_board_fen, str):
        return None

    move_ucis = _string_list(clip.get("move_ucis"))
    move_frame_indices = _int_list(clip.get("move_frame_indices"))
    if len(move_ucis) != len(move_frame_indices):
        return None

    frame_indices = _int_list(clip.get("frame_indices"))
    if len(frame_indices) != int(frames.shape[0]):
        frame_indices = list(range(int(frames.shape[0])))
    frame_to_sample_index = {frame_index: index for index, frame_index in enumerate(frame_indices)}

    moves_by_sample_index: dict[int, list[str]] = defaultdict(list)
    for frame_index, move_uci in zip(move_frame_indices, move_ucis):
        sample_index = frame_to_sample_index.get(frame_index)
        if sample_index is None:
            continue
        moves_by_sample_index[sample_index].append(move_uci)
    if any(len(moves) > 1 for moves in moves_by_sample_index.values()):
        return None

    vocab = get_vocabulary()
    board = _build_replay_board(initial_board_fen, move_ucis[0] if move_ucis else None)
    move_targets: list[int] = []
    detect_targets: list[float] = []
    legal_masks: list[torch.Tensor] = []
    board_fens: list[str] = []

    total_frames = int(frames.shape[0])
    for sample_index in range(total_frames):
        legal_masks.append(get_legal_mask(board))
        moves = moves_by_sample_index.get(sample_index)
        if not moves:
            move_targets.append(NO_MOVE_IDX)
            detect_targets.append(0.0)
            board_fens.append(board.fen())
            continue

        move_uci = moves[0]
        if not vocab.contains(move_uci):
            return None
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None
        move_targets.append(vocab.uci_to_index(move_uci))
        detect_targets.append(1.0)
        board.push(move)
        board_fens.append(board.fen())

    return (
        torch.tensor(move_targets, dtype=torch.long),
        torch.tensor(detect_targets, dtype=torch.float32),
        torch.stack(legal_masks, dim=0),
        board_fens,
    )


def _infer_annotated_moves(
    clip_rows: list[Any],
    initial_board_fen: str,
) -> tuple[list[int], list[float]]:
    vocab = get_vocabulary()
    board_hypotheses = build_board_hypotheses_from_piece_fen(initial_board_fen)
    current_board = board_hypotheses[0]

    move_targets: list[int] = []
    detect_targets: list[float] = []
    for row_index, row in enumerate(clip_rows):
        target_board_fen = _labels_to_board_fen(row.labels)
        if target_board_fen is None:
            move_targets.append(NO_MOVE_IDX)
            detect_targets.append(0.0)
            continue
        if row_index == 0:
            if current_board.board_fen() != target_board_fen:
                current_board = _best_matching_hypothesis(board_hypotheses, target_board_fen)
            move_targets.append(NO_MOVE_IDX)
            detect_targets.append(0.0)
            continue
        if current_board.board_fen() == target_board_fen:
            move_targets.append(NO_MOVE_IDX)
            detect_targets.append(0.0)
            continue
        move = find_move_between_positions(current_board, target_board_fen)
        if move is None:
            fallback_board = current_board.copy(stack=False)
            fallback_board.turn = not fallback_board.turn
            move = find_move_between_positions(fallback_board, target_board_fen)
            if move is not None:
                current_board.turn = fallback_board.turn
        if move is None or not vocab.contains(move.uci()):
            move_targets.append(NO_MOVE_IDX)
            detect_targets.append(0.0)
            continue
        move_targets.append(vocab.uci_to_index(move.uci()))
        detect_targets.append(1.0)
        current_board.push(move)
    return move_targets, detect_targets


def build_board_hypotheses_from_piece_fen(initial_board_fen: str) -> list[chess.Board]:
    white_board = chess.Board()
    white_board.set_board_fen(initial_board_fen)
    white_board.turn = chess.WHITE
    black_board = white_board.copy(stack=False)
    black_board.turn = chess.BLACK
    return [white_board, black_board]


def _best_matching_hypothesis(hypotheses: list[chess.Board], target_board_fen: str) -> chess.Board:
    for board in hypotheses:
        if board.board_fen() == target_board_fen:
            return board.copy(stack=False)
    return hypotheses[0].copy(stack=False)


def _labels_to_board_fen(labels: tuple[int, ...]) -> str | None:
    if len(labels) != 64:
        return None
    ranks: list[str] = []
    for row in range(8):
        empty_run = 0
        rank_parts: list[str] = []
        for col in range(8):
            label = labels[row * 8 + col]
            if label == 0:
                empty_run += 1
                continue
            if empty_run:
                rank_parts.append(str(empty_run))
                empty_run = 0
            rank_parts.append(_CLASS_NAMES[label])
        if empty_run:
            rank_parts.append(str(empty_run))
        ranks.append("".join(rank_parts) or "8")
    return "/".join(ranks)


def _initial_board_fen_for_clip(clip_path: str) -> str:
    clip = torch.load(_PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
    if not isinstance(clip, dict):
        raise ValueError(f"Invalid clip payload: {clip_path}")
    initial_board_fen = clip.get("initial_board_fen")
    if not isinstance(initial_board_fen, str):
        raise ValueError(f"Clip is missing initial_board_fen: {clip_path}")
    return initial_board_fen


_CLASS_NAMES = [
    "empty",
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
]
