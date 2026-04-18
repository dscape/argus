"""Physical move-model datasets built from real clip metadata and held-out annotations."""

from __future__ import annotations

import logging
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
from pipeline.physical.shared import splits
from pipeline.physical.piece_projection import extract_board_neighborhood_crop
from pipeline.physical.shared.annotation_rows import (
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.shared.real_board_data import (
    _build_replay_board,
    _frame_tensor_to_rgb,
    _int_list,
    _string_list,
    load_real_board_rows,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_DEFAULT_EVAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"
_DEFAULT_IMAGE_SIZE = 224
_OBSERVATION_MODE = "piece_projection_board"
_DEFAULT_BOARD_CROP_MARGIN = 0.18


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
    initial_side_to_move: str | None
    frames: torch.Tensor
    board_corners: torch.Tensor | None
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
    positive_window_repeat: int = 1,
    selection_source_video_ids: set[str] | None = None,
    exclude_selection_source_video_ids: bool = False,
    observation_mode: str = _OBSERVATION_MODE,
    move_target_pre_frames: int = 0,
    detect_target_radius: int = 0,
    detect_target_decay: float = 0.5,
    board_crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
) -> tuple[list[dict[str, Any]], list[PhysicalMoveWindowMetadata]]:
    """Build fixed-length real physical move windows from replay metadata."""
    splits.ensure_annotation_layout_migrated()
    if clip_length <= 0:
        raise ValueError(f"clip_length must be > 0, got {clip_length}")
    if negative_window_stride <= 0:
        raise ValueError(f"negative_window_stride must be > 0, got {negative_window_stride}")
    if observation_mode != _OBSERVATION_MODE:
        raise ValueError(f"Unsupported observation_mode: {observation_mode}")
    if positive_window_repeat <= 0:
        raise ValueError(f"positive_window_repeat must be > 0, got {positive_window_repeat}")
    if move_target_pre_frames < 0:
        raise ValueError(f"move_target_pre_frames must be >= 0, got {move_target_pre_frames}")
    if detect_target_radius < 0:
        raise ValueError(f"detect_target_radius must be >= 0, got {detect_target_radius}")
    if not 0.0 <= detect_target_decay <= 1.0:
        raise ValueError(f"detect_target_decay must be in [0, 1], got {detect_target_decay}")

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
            observation_mode=observation_mode,
            move_target_pre_frames=move_target_pre_frames,
            detect_target_radius=detect_target_radius,
            detect_target_decay=detect_target_decay,
            board_crop_margin=board_crop_margin,
        )
        if clip_data is None:
            continue

        clip_windows = _slice_move_windows(
            clip_data,
            clip_length=clip_length,
            negative_window_stride=negative_window_stride,
            max_negative_windows=max_negative_windows_per_clip,
            positive_window_repeat=positive_window_repeat,
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


def load_real_move_sequences(
    *,
    clips_dir: str | Path = _DEFAULT_CLIPS_DIR,
    eval_root: str | Path = _DEFAULT_EVAL_ROOT,
    image_size: int = _DEFAULT_IMAGE_SIZE,
    selection_source_video_ids: set[str] | None = None,
    exclude_selection_source_video_ids: bool = False,
    observation_mode: str = _OBSERVATION_MODE,
    board_crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
) -> list[PhysicalEvalMoveSequence]:
    """Load full real replay clips as sequence-selection inputs."""
    splits.ensure_annotation_layout_migrated()
    if observation_mode != _OBSERVATION_MODE:
        raise ValueError(f"Unsupported observation_mode: {observation_mode}")

    rows = load_real_board_rows(
        clips_dir=clips_dir,
        eval_root=eval_root,
        frame_stride=1,
        max_frames=None,
    )
    rows_by_clip: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        rows_by_clip[row.clip_path].append(row)

    sequences: list[PhysicalEvalMoveSequence] = []
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
            observation_mode=observation_mode,
            move_target_pre_frames=0,
            detect_target_radius=0,
            detect_target_decay=0.5,
            board_crop_margin=board_crop_margin,
        )
        if clip_data is None:
            continue

        exact_replay_targets = _replay_targets_for_clip(clip_payload)
        initial_board_fen = clip_payload.get("initial_board_fen")
        if exact_replay_targets is None or not isinstance(initial_board_fen, str):
            continue
        exact_move_targets, exact_detect_targets, _legal_masks, _board_fens = exact_replay_targets

        initial_side_to_move = clip_payload.get("initial_side_to_move")
        if not isinstance(initial_side_to_move, str):
            initial_side_to_move = None

        labels = tuple(tuple(int(value) for value in row.labels) for row in clip_rows)
        frame_indices = tuple(int(row.frame_index) for row in clip_rows)
        board_corners = clip_data.get("board_corners")
        sequence_frames = clip_data.get("frames")
        if not isinstance(sequence_frames, torch.Tensor):
            continue
        if len(labels) != int(sequence_frames.shape[0]):
            continue

        sequences.append(
            PhysicalEvalMoveSequence(
                clip_path=clip_path,
                source_video_id=source_video_id,
                initial_board_fen=initial_board_fen,
                initial_side_to_move=initial_side_to_move,
                frames=sequence_frames.clone(),
                board_corners=(
                    None
                    if not isinstance(board_corners, torch.Tensor)
                    else board_corners.clone()
                ),
                frame_indices=frame_indices,
                labels=labels,
                inferred_move_targets=tuple(int(value) for value in exact_move_targets.tolist()),
                inferred_detect_targets=tuple(
                    float(value) for value in exact_detect_targets.tolist()
                ),
            )
        )

    sequences.sort(key=lambda sequence: sequence.clip_path)
    return sequences


def load_eval_move_sequences(
    *,
    annotation_root: str | Path = _DEFAULT_EVAL_ROOT,
    image_size: int = _DEFAULT_IMAGE_SIZE,
    observation_mode: str = _OBSERVATION_MODE,
    board_crop_margin: float = _DEFAULT_BOARD_CROP_MARGIN,
) -> list[PhysicalEvalMoveSequence]:
    """Load held-out annotated sequences for move-model evaluation."""
    if observation_mode != _OBSERVATION_MODE:
        raise ValueError(f"Unsupported observation_mode: {observation_mode}")

    rows = load_annotated_oblique_rows(annotation_root)
    rows_by_clip: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        rows_by_clip[row.clip_path].append(row)

    clip_cache: dict[Path, dict[str, object]] = {}
    sequences: list[PhysicalEvalMoveSequence] = []
    for clip_path, clip_rows in rows_by_clip.items():
        clip_rows.sort(key=lambda row: row.frame_index)
        initial_board_fen, initial_side_to_move = _initial_board_state_for_clip(clip_path)
        inference = _infer_annotated_moves(
            clip_rows,
            initial_board_fen,
            initial_side_to_move=initial_side_to_move,
        )

        frames = []
        labels = []
        frame_indices = []
        board_corners = []
        for row in clip_rows:
            image_bgr = _load_clip_frame_bgr(row, clip_cache=clip_cache)
            board_image, scaled_corners = _prepare_projected_board_frame(
                image_bgr,
                row.corners,
                image_size=image_size,
                crop_margin=board_crop_margin,
            )
            frames.append(board_image)
            board_corners.append(scaled_corners)
            labels.append(tuple(int(value) for value in row.labels))
            frame_indices.append(int(row.frame_index))

        sequences.append(
            PhysicalEvalMoveSequence(
                clip_path=clip_path,
                source_video_id=clip_rows[0].source_video_id,
                initial_board_fen=initial_board_fen,
                initial_side_to_move=initial_side_to_move,
                frames=torch.stack(frames, dim=0),
                board_corners=torch.stack(board_corners, dim=0),
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
    observation_mode: str,
    move_target_pre_frames: int,
    detect_target_radius: int,
    detect_target_decay: float,
    board_crop_margin: float,
) -> dict[str, Any] | None:
    frames = clip.get("frames")
    initial_board_fen = clip.get("initial_board_fen")
    if not isinstance(frames, torch.Tensor) or not isinstance(initial_board_fen, str):
        return None

    # Warn about legacy 224x224 downscaled clips when we still consume clip-native frames.
    if frames.shape[-1] == 224 and frames.shape[-2] == 224:
        logger.warning(
            "Clip %s has degraded 224x224 frames — regenerate for native resolution",
            clip_path,
        )

    corners = clip_rows[0].corners
    exact_replay_targets = _replay_targets_for_clip(clip)
    if exact_replay_targets is None:
        return None
    replay_targets = _build_replay_supervision_targets(
        clip,
        move_target_pre_frames=move_target_pre_frames,
        detect_target_radius=detect_target_radius,
        detect_target_decay=detect_target_decay,
    )
    if replay_targets is None:
        return None
    (
        move_targets,
        detect_targets,
        legal_masks,
        board_fens,
        move_loss_mask,
        move_loss_weights,
    ) = replay_targets
    if int(frames.shape[0]) != len(move_targets):
        return None

    if observation_mode != _OBSERVATION_MODE:
        raise ValueError(f"Unsupported observation_mode: {observation_mode}")

    clip_frames: list[torch.Tensor] = []
    board_corners: list[torch.Tensor] = []
    for frame_index in range(int(frames.shape[0])):
        image_bgr = cv2.cvtColor(_frame_tensor_to_rgb(frames[frame_index]), cv2.COLOR_RGB2BGR)
        board_image, scaled_corners = _prepare_projected_board_frame(
            image_bgr,
            corners,
            image_size=image_size,
            crop_margin=board_crop_margin,
        )
        clip_frames.append(board_image)
        board_corners.append(scaled_corners)

    payload: dict[str, Any] = {
        "clip_path": clip_path,
        "frames": torch.stack(clip_frames, dim=0),
        "move_targets": move_targets,
        "detect_targets": detect_targets,
        "legal_masks": legal_masks,
        "move_mask": detect_targets > 0.5,
        "exact_move_mask": exact_replay_targets[1] > 0.5,
        "move_loss_mask": move_loss_mask,
        "move_loss_weights": move_loss_weights,
        "fens": board_fens,
    }
    payload["board_corners"] = torch.stack(board_corners, dim=0)
    return payload


def _slice_move_windows(
    clip_data: dict[str, Any],
    *,
    clip_length: int,
    negative_window_stride: int,
    max_negative_windows: int,
    positive_window_repeat: int,
) -> list[tuple[dict[str, Any], int, int, bool]]:
    frames = clip_data["frames"]
    move_targets = clip_data["move_targets"]
    detect_targets = clip_data["detect_targets"]
    legal_masks = clip_data["legal_masks"]
    board_fens = clip_data.get("fens")
    board_corners = clip_data.get("board_corners")
    move_loss_mask = clip_data.get("move_loss_mask")
    move_loss_weights = clip_data.get("move_loss_weights")
    exact_move_mask = clip_data.get("exact_move_mask")
    total_frames = int(frames.shape[0])
    if total_frames == 0:
        return []

    window_starts: dict[int, bool] = {}
    positive_window_mask = (
        exact_move_mask if isinstance(exact_move_mask, torch.Tensor) else detect_targets > 0.5
    )
    move_indices = torch.nonzero(positive_window_mask, as_tuple=False).reshape(-1).tolist()
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
        if isinstance(board_corners, torch.Tensor):
            payload["board_corners"] = board_corners[window_start:window_end].clone()
        if isinstance(move_loss_mask, torch.Tensor):
            payload["move_loss_mask"] = move_loss_mask[window_start:window_end].clone()
        if isinstance(move_loss_weights, torch.Tensor):
            payload["move_loss_weights"] = move_loss_weights[window_start:window_end].clone()
        if isinstance(exact_move_mask, torch.Tensor):
            payload["exact_move_mask"] = exact_move_mask[window_start:window_end].clone()
        repeat_count = positive_window_repeat if contains_move else 1
        for _ in range(repeat_count):
            windows.append((payload, window_start, window_end, contains_move))
    return windows


def _build_replay_supervision_targets(
    clip: dict[str, Any],
    *,
    move_target_pre_frames: int,
    detect_target_radius: int,
    detect_target_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor] | None:
    replay_targets = _replay_targets_for_clip(clip)
    if replay_targets is None:
        return None
    move_targets, detect_targets, legal_masks, board_fens = replay_targets

    tolerant_move_targets = torch.full_like(move_targets, NO_MOVE_IDX)
    move_loss_mask = torch.zeros_like(move_targets, dtype=torch.bool)
    move_loss_weights = torch.zeros_like(detect_targets, dtype=torch.float32)
    tolerant_detect_targets = detect_targets.clone()

    move_indices = torch.nonzero(detect_targets > 0.5, as_tuple=False).reshape(-1).tolist()
    for move_index in move_indices:
        target_move = int(move_targets[move_index].item())
        for offset in range(move_target_pre_frames + 1):
            frame_index = move_index - offset
            if frame_index < 0:
                break
            weight = detect_target_decay**offset if offset > 0 else 1.0
            if weight <= float(move_loss_weights[frame_index].item()):
                continue
            tolerant_move_targets[frame_index] = target_move
            move_loss_mask[frame_index] = True
            move_loss_weights[frame_index] = weight
        for offset in range(-detect_target_radius, detect_target_radius + 1):
            frame_index = move_index + offset
            if frame_index < 0 or frame_index >= len(tolerant_detect_targets):
                continue
            weight = detect_target_decay ** abs(offset) if offset != 0 else 1.0
            tolerant_detect_targets[frame_index] = torch.maximum(
                tolerant_detect_targets[frame_index],
                torch.tensor(weight, dtype=tolerant_detect_targets.dtype),
            )

    return (
        tolerant_move_targets,
        tolerant_detect_targets,
        legal_masks,
        board_fens,
        move_loss_mask,
        move_loss_weights,
    )


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
    initial_side_to_move = clip.get("initial_side_to_move")
    side_to_move = initial_side_to_move if isinstance(initial_side_to_move, str) else None
    board = _build_replay_board(
        initial_board_fen,
        move_ucis[0] if move_ucis else None,
        initial_side_to_move=side_to_move,
    )
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
    *,
    initial_side_to_move: str | None = None,
) -> tuple[list[int], list[float]]:
    vocab = get_vocabulary()
    board_hypotheses = build_board_hypotheses_from_piece_fen(
        initial_board_fen,
        initial_side_to_move=initial_side_to_move,
    )
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


def build_board_hypotheses_from_piece_fen(
    initial_board_fen: str,
    *,
    initial_side_to_move: str | None = None,
) -> list[chess.Board]:
    if initial_side_to_move in {"w", "b"}:
        board = chess.Board()
        board.set_board_fen(initial_board_fen)
        board.turn = chess.WHITE if initial_side_to_move == "w" else chess.BLACK
        return [board]

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


def _prepare_projected_board_frame(
    image_bgr: Any,
    corners: tuple[tuple[float, float], ...] | list[list[float]],
    *,
    image_size: int,
    crop_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    crop = extract_board_neighborhood_crop(image_bgr, corners, crop_margin=crop_margin)
    rgb = cv2.cvtColor(crop.image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= image_size else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=interpolation)
    image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    height, width = crop.image_bgr.shape[:2]
    scaled_corners = crop.corners.copy()
    scaled_corners[:, 0] *= float(image_size) / max(float(width), 1.0)
    scaled_corners[:, 1] *= float(image_size) / max(float(height), 1.0)
    return image_tensor, torch.from_numpy(scaled_corners.astype("float32"))


def _initial_board_state_for_clip(clip_path: str) -> tuple[str, str | None]:
    clip = torch.load(_PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
    if not isinstance(clip, dict):
        raise ValueError(f"Invalid clip payload: {clip_path}")
    initial_board_fen = clip.get("initial_board_fen")
    if not isinstance(initial_board_fen, str):
        raise ValueError(f"Clip is missing initial_board_fen: {clip_path}")
    raw_side_to_move = clip.get("initial_side_to_move")
    side_to_move = raw_side_to_move if isinstance(raw_side_to_move, str) else None
    return initial_board_fen, side_to_move


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
