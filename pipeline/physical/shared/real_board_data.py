"""Real physical-board training data built from replayed clip metadata."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from argus.chess.board_state import fen_to_square_targets
from pipeline.physical.board_probe.board_data import (
    INPUT_SIZE,
    preprocess_board_neighborhood_geometry,
)
from pipeline.physical.shared import splits
from pipeline.physical.shared.eval_dataset import (
    DEFAULT_BOARD_SIZE,
    get_held_out_source_video_ids,
    rectify_board_image,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CLIPS_DIR = _PROJECT_ROOT / "data" / "argus" / "train_real"
_DEFAULT_EVAL_ROOT = _PROJECT_ROOT / "data" / "physical" / "val"
_REAL_CLIP_RE = re.compile(r"^clip_[^_]+_(?P<video_id>[^_]+)_.*\.pt$")


@dataclass(frozen=True)
class PhysicalRealBoardRow:
    clip_path: str
    frame_index: int
    source_video_id: str | None
    source_channel_handle: str | None
    corners: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]


class PhysicalRealBoardDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Replay-derived board neighborhoods with projected piece-box geometry."""

    def __init__(
        self,
        *,
        clips_dir: str | Path = _DEFAULT_CLIPS_DIR,
        eval_root: str | Path = _DEFAULT_EVAL_ROOT,
        image_size: int = INPUT_SIZE,
        rows: list[PhysicalRealBoardRow] | None = None,
        frame_stride: int = 4,
        max_frames: int | None = None,
        seed: int = 42,
        exclude_move_neighborhood: int = -1,
    ) -> None:
        self.clips_dir = Path(clips_dir)
        self.eval_root = Path(eval_root)
        self.image_size = image_size
        self.rows = rows or load_real_board_rows(
            clips_dir=self.clips_dir,
            eval_root=self.eval_root,
            frame_stride=frame_stride,
            max_frames=max_frames,
            seed=seed,
            exclude_move_neighborhood=exclude_move_neighborhood,
        )
        self._clip_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        clip = self._load_clip(_PROJECT_ROOT / row.clip_path)
        frames = clip.get("frames")
        if not isinstance(frames, torch.Tensor):
            raise ValueError(f"Clip has no frames tensor: {row.clip_path}")
        frame = frames[row.frame_index]
        image_bgr = cv2.cvtColor(_frame_tensor_to_rgb(frame), cv2.COLOR_RGB2BGR)
        image, _scaled_corners, piece_bboxes = preprocess_board_neighborhood_geometry(
            image_bgr,
            row.corners,
            size=self.image_size,
        )
        targets = torch.tensor(row.labels, dtype=torch.long)
        return image, targets, piece_bboxes

    def _load_clip(self, clip_path: Path) -> dict[str, Any]:
        cached = self._clip_cache.get(clip_path)
        if cached is not None:
            return cached
        clip = torch.load(clip_path, map_location="cpu", weights_only=False)
        if not isinstance(clip, dict):
            raise ValueError(f"Invalid real clip: {clip_path}")
        self._clip_cache[clip_path] = clip
        return clip


def load_real_board_rows(
    *,
    clips_dir: str | Path = _DEFAULT_CLIPS_DIR,
    eval_root: str | Path = _DEFAULT_EVAL_ROOT,
    frame_stride: int = 4,
    max_frames: int | None = None,
    seed: int = 42,
    refine_corners: bool = True,
    exclude_move_neighborhood: int = -1,
) -> list[PhysicalRealBoardRow]:
    splits.ensure_annotation_layout_migrated()
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be > 0, got {frame_stride}")
    if exclude_move_neighborhood < -1:
        raise ValueError(
            f"exclude_move_neighborhood must be >= -1, got {exclude_move_neighborhood}"
        )

    clip_dir_path = Path(clips_dir)
    if not clip_dir_path.is_absolute():
        clip_dir_path = (_PROJECT_ROOT / clip_dir_path).resolve()
    channel_templates = infer_channel_corner_templates(eval_root=eval_root)
    held_out_source_video_ids = set(get_held_out_source_video_ids())
    rows: list[PhysicalRealBoardRow] = []

    for clip_path in sorted(clip_dir_path.glob("clip_*.pt")):
        clip = torch.load(clip_path, map_location="cpu", weights_only=False)
        if not isinstance(clip, dict):
            continue

        source_video_id = _source_video_id_from_clip(clip, clip_path)
        if source_video_id in held_out_source_video_ids:
            continue

        source_channel_handle = clip.get("source_channel_handle")
        if not isinstance(source_channel_handle, str):
            continue
        corners = channel_templates.get(source_channel_handle)
        if corners is None:
            continue
        if refine_corners:
            corners = _refine_clip_corners(clip, corners)

        fens = replay_clip_display_fens(clip)
        move_sample_indices = replay_clip_move_sample_indices(clip)
        excluded_sample_indices = build_excluded_move_neighborhood(
            move_sample_indices,
            total_frames=len(fens),
            neighborhood=exclude_move_neighborhood,
        )
        for frame_index, fen in enumerate(fens):
            if fen is None or frame_index in excluded_sample_indices:
                continue
            if frame_index % frame_stride != 0 and frame_index not in move_sample_indices:
                continue
            labels = tuple(int(value) for value in fen_to_square_targets(fen).tolist())
            rows.append(
                PhysicalRealBoardRow(
                    clip_path=str(clip_path.resolve().relative_to(_PROJECT_ROOT.resolve())),
                    frame_index=frame_index,
                    source_video_id=source_video_id,
                    source_channel_handle=source_channel_handle,
                    corners=corners,
                    labels=labels,
                )
            )

    if max_frames is not None and len(rows) > max_frames:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_frames]

    return rows


def infer_channel_corner_templates(
    *,
    eval_root: str | Path = _DEFAULT_EVAL_ROOT,
) -> dict[str, tuple[tuple[float, float], ...]]:
    splits.ensure_annotation_layout_migrated()
    annotations_path = Path(eval_root) / "board_annotations.jsonl"
    if not annotations_path.exists():
        return {}

    clip_channel_cache: dict[str, str | None] = {}
    corners_by_channel: dict[str, list[np.ndarray]] = {}
    for line in annotations_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        clip_path = payload.get("clip_path")
        raw_corners = payload.get("corners")
        if (
            not isinstance(clip_path, str)
            or not isinstance(raw_corners, list)
            or len(raw_corners) != 4
        ):
            continue

        source_channel_handle = clip_channel_cache.get(clip_path)
        if source_channel_handle is None and clip_path not in clip_channel_cache:
            clip = torch.load(_PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
            handle = clip.get("source_channel_handle") if isinstance(clip, dict) else None
            source_channel_handle = handle if isinstance(handle, str) else None
            clip_channel_cache[clip_path] = source_channel_handle

        if not isinstance(source_channel_handle, str):
            continue

        corners_by_channel.setdefault(source_channel_handle, []).append(
            np.asarray(raw_corners, dtype=np.float32)
        )

    templates: dict[str, tuple[tuple[float, float], ...]] = {}
    for channel_handle, channel_corners in corners_by_channel.items():
        median_corners = np.median(np.stack(channel_corners), axis=0)
        templates[channel_handle] = tuple(
            (float(point[0]), float(point[1])) for point in median_corners
        )
    return templates


def replay_clip_display_fens(clip: dict[str, Any]) -> list[str | None]:
    frames = clip.get("frames")
    if not isinstance(frames, torch.Tensor):
        return []

    initial_board_fen = clip.get("initial_board_fen")
    if not isinstance(initial_board_fen, str):
        return [None] * int(frames.shape[0])

    move_ucis = _string_list(clip.get("move_ucis"))
    move_frame_indices = _int_list(clip.get("move_frame_indices"))
    if len(move_ucis) != len(move_frame_indices):
        return [None] * int(frames.shape[0])

    sampled_frame_indices = sampled_clip_frame_indices(clip)
    if len(sampled_frame_indices) != int(frames.shape[0]):
        return [None] * int(frames.shape[0])

    moves_by_sample_index: dict[int, list[str]] = {}
    frame_to_sample_index = {
        frame_index: index for index, frame_index in enumerate(sampled_frame_indices)
    }
    for frame_index, move_uci in zip(move_frame_indices, move_ucis):
        sample_index = frame_to_sample_index.get(frame_index)
        if sample_index is None:
            return [None] * int(frames.shape[0])
        moves_by_sample_index.setdefault(sample_index, []).append(move_uci)

    initial_side_to_move = clip.get("initial_side_to_move")
    side_to_move = initial_side_to_move if isinstance(initial_side_to_move, str) else None
    board = _build_replay_board(
        initial_board_fen,
        move_ucis[0] if move_ucis else None,
        initial_side_to_move=side_to_move,
    )
    frame_fens: list[str | None] = []
    total_frames = int(frames.shape[0])
    for sample_index in range(total_frames):
        moves = moves_by_sample_index.get(sample_index)
        if moves is None:
            frame_fens.append(board.fen())
            continue

        try:
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    raise ValueError(f"Illegal move at frame {sample_index}: {move_uci}")
                board.push(move)
        except ValueError:
            frame_fens.append(None)
            continue
        frame_fens.append(board.fen())

    return frame_fens


def replay_clip_move_sample_indices(clip: dict[str, Any]) -> set[int]:
    sampled_frame_indices = sampled_clip_frame_indices(clip)
    move_frame_indices = set(_int_list(clip.get("move_frame_indices")))
    return {
        sample_index
        for sample_index, frame_index in enumerate(sampled_frame_indices)
        if frame_index in move_frame_indices
    }


def build_excluded_move_neighborhood(
    move_sample_indices: set[int],
    *,
    total_frames: int,
    neighborhood: int,
) -> set[int]:
    if neighborhood < 0:
        return set()

    excluded: set[int] = set()
    for move_sample_index in move_sample_indices:
        for offset in range(-neighborhood, neighborhood + 1):
            sample_index = move_sample_index + offset
            if 0 <= sample_index < total_frames:
                excluded.add(sample_index)
    return excluded


def sampled_clip_frame_indices(clip: dict[str, Any]) -> list[int]:
    frames = clip.get("frames")
    if not isinstance(frames, torch.Tensor):
        return []
    sampled_frame_indices = _int_list(clip.get("frame_indices"))
    if len(sampled_frame_indices) == int(frames.shape[0]):
        return sampled_frame_indices
    return list(range(int(frames.shape[0])))


def _refine_clip_corners(
    clip: dict[str, Any],
    initial_corners: tuple[tuple[float, float], ...],
    *,
    max_offset: int = 12,
) -> tuple[tuple[float, float], ...]:
    frames = clip.get("frames")
    if not isinstance(frames, torch.Tensor) or frames.shape[0] == 0:
        return initial_corners

    frame = _frame_tensor_to_rgb(frames[0])
    corners = np.asarray(initial_corners, dtype=np.float32)
    initial = corners.copy()
    height, width = frame.shape[:2]
    best_score = _rectified_board_score(
        rectify_board_image(frame, corners.tolist(), output_size=256)
    )

    for step in (4, 2, 1):
        improved = True
        while improved:
            improved = False
            for corner_index in range(4):
                current = corners[corner_index].copy()
                local_best_score = best_score
                local_best_corner = current.copy()
                for dx in (-step, 0, step):
                    for dy in (-step, 0, step):
                        candidate = corners.copy()
                        candidate[corner_index] = current + np.array([dx, dy], dtype=np.float32)
                        delta = candidate[corner_index] - initial[corner_index]
                        if np.max(np.abs(delta)) > max_offset:
                            continue
                        if not _corners_are_valid(candidate, width=width, height=height):
                            continue
                        candidate_score = _rectified_board_score(
                            rectify_board_image(frame, candidate.tolist(), output_size=256)
                        )
                        if candidate_score > local_best_score:
                            local_best_score = candidate_score
                            local_best_corner = candidate[corner_index].copy()
                if local_best_score > best_score:
                    corners[corner_index] = local_best_corner
                    best_score = local_best_score
                    improved = True

    return tuple((float(point[0]), float(point[1])) for point in corners)


def _rectified_board_score(board_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    crop = gray[12:-12, 12:-12]
    cell_height = crop.shape[0] // 8
    cell_width = crop.shape[1] // 8
    cells = crop[: cell_height * 8, : cell_width * 8].reshape(8, cell_height, 8, cell_width)
    cells = cells.transpose(0, 2, 1, 3)
    means = cells.mean(axis=(2, 3))
    stds = cells.std(axis=(2, 3))
    mask = (np.indices((8, 8)).sum(axis=0) % 2) == 0
    return max(
        _orientation_score(means, stds, mask),
        _orientation_score(means, stds, ~mask),
    )


def _orientation_score(means: np.ndarray, stds: np.ndarray, mask: np.ndarray) -> float:
    light = means[mask]
    dark = means[~mask]
    return float(
        abs(light.mean() - dark.mean()) - 0.75 * (light.std() + dark.std()) - 0.25 * stds.mean()
    )


def _corners_are_valid(corners: np.ndarray, *, width: int, height: int) -> bool:
    if (
        (corners[:, 0] < 0).any()
        or (corners[:, 0] > width - 1).any()
        or (corners[:, 1] < 0).any()
        or (corners[:, 1] > height - 1).any()
    ):
        return False

    area = 0.0
    for index in range(4):
        x1, y1 = corners[index]
        x2, y2 = corners[(index + 1) % 4]
        area += x1 * y2 - x2 * y1
    return abs(area) > 100.0


def _build_replay_board(
    initial_board_fen: str,
    first_move_uci: str | None,
    *,
    initial_side_to_move: str | None = None,
) -> chess.Board:
    board = chess.Board()
    board.set_board_fen(initial_board_fen)
    if initial_side_to_move in {"w", "b"}:
        board.turn = chess.WHITE if initial_side_to_move == "w" else chess.BLACK
        return board
    if first_move_uci is None:
        return board

    try:
        first_move = chess.Move.from_uci(first_move_uci)
    except ValueError:
        return board

    white_board = board.copy(stack=False)
    white_board.turn = chess.WHITE
    black_board = board.copy(stack=False)
    black_board.turn = chess.BLACK
    if first_move in black_board.legal_moves and first_move not in white_board.legal_moves:
        return black_board
    return white_board


def _frame_tensor_to_rgb(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor with 3 dims, got {tuple(frame.shape)}")
    if frame.shape[0] == 3:
        chw = frame
    elif frame.shape[-1] == 3:
        chw = frame.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB frame tensor, got {tuple(frame.shape)}")

    if chw.dtype == torch.uint8:
        array = chw.permute(1, 2, 0).cpu().numpy()
        return array.astype(np.uint8)

    rgb = chw.to(torch.float32)
    if float(rgb.max().item()) <= 1.0:
        rgb = rgb * 255.0
    array = rgb.clamp(0.0, 255.0).permute(1, 2, 0).cpu().numpy()
    return array.astype(np.uint8)


def _source_video_id_from_clip(clip: dict[str, Any], clip_path: Path) -> str | None:
    source_video_id = clip.get("source_video_id")
    if isinstance(source_video_id, str):
        return source_video_id
    match = _REAL_CLIP_RE.match(clip_path.name)
    return None if match is None else str(match.group("video_id"))


def _int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.tolist()]
    if isinstance(value, list):
        return [int(item) for item in value]
    return []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]
