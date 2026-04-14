"""PyTorch Dataset for Argus training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from argus.chess.board_state import fen_to_square_targets

logger = logging.getLogger(__name__)


class ArgusDataset(Dataset):  # type: ignore[type-arg]
    """Dataset for Argus chess move detection training.

    Loads pre-generated clips from disk (produced by synth2d.generate_dataset
    or a real-data pipeline). Each clip is a dict with tensors for frames,
    move targets, detection targets, legal masks, and move masks.
    """

    def __init__(
        self,
        data_dir: str | Path,
        clip_length: int = 16,
        transform: Any | None = None,
        max_clips: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.clip_length = clip_length
        self.transform = transform

        self.clip_paths: list[Path] = sorted(self.data_dir.glob("clip_*.pt"))
        if max_clips is not None:
            self.clip_paths = self.clip_paths[:max_clips]

        if len(self.clip_paths) == 0:
            logger.warning(f"No clip files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.clip_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clip_data = torch.load(self.clip_paths[idx], map_location="cpu", weights_only=True)
        return _prepare_sample(clip_data, clip_length=self.clip_length, transform=self.transform)


class ArgusInMemoryDataset(Dataset):  # type: ignore[type-arg]
    """In-memory dataset from pre-generated clips (e.g., from synth2d)."""

    def __init__(
        self,
        clips: list[dict[str, Any]],
        clip_length: int = 16,
        transform: Any | None = None,
    ) -> None:
        self.clips = clips
        self.clip_length = clip_length
        self.transform = transform

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return _prepare_sample(
            self.clips[idx], clip_length=self.clip_length, transform=self.transform
        )


def _prepare_sample(
    clip_data: dict[str, Any],
    *,
    clip_length: int,
    transform: Any | None,
) -> dict[str, torch.Tensor]:
    frames = _prepare_frames(clip_data["frames"])
    move_targets = clip_data["move_targets"].to(torch.long)
    detect_targets = clip_data["detect_targets"].to(torch.float32)
    legal_masks = clip_data["legal_masks"].to(torch.bool)
    square_targets = _build_square_targets(clip_data)
    board_corners = _optional_tensor(clip_data.get("board_corners"), dtype=torch.float32)
    move_loss_mask = _optional_tensor(clip_data.get("move_loss_mask"), dtype=torch.bool)
    move_loss_weights = _optional_tensor(clip_data.get("move_loss_weights"), dtype=torch.float32)

    if transform is not None:
        frames = transform(frames)

    sequence_length = frames.shape[0]
    if sequence_length < clip_length:
        frames = _pad_tensor(frames, clip_length, dim=0)
        move_targets = _pad_tensor(move_targets, clip_length, dim=0)
        detect_targets = _pad_tensor(detect_targets, clip_length, dim=0)
        legal_masks = _pad_tensor(legal_masks, clip_length, dim=0)
        if square_targets is not None:
            square_targets = _pad_tensor(square_targets, clip_length, dim=0, fill_value=-100)
        if board_corners is not None:
            board_corners = _pad_temporal_tensor_with_last(board_corners, clip_length)
        if move_loss_mask is not None:
            move_loss_mask = _pad_tensor(move_loss_mask, clip_length, dim=0, fill_value=False)
        if move_loss_weights is not None:
            move_loss_weights = _pad_tensor(move_loss_weights, clip_length, dim=0, fill_value=0.0)
    elif sequence_length > clip_length:
        frames = frames[:clip_length]
        move_targets = move_targets[:clip_length]
        detect_targets = detect_targets[:clip_length]
        legal_masks = legal_masks[:clip_length]
        if square_targets is not None:
            square_targets = square_targets[:clip_length]
        if board_corners is not None:
            board_corners = board_corners[:clip_length]
        if move_loss_mask is not None:
            move_loss_mask = move_loss_mask[:clip_length]
        if move_loss_weights is not None:
            move_loss_weights = move_loss_weights[:clip_length]

    move_mask = _derive_move_mask(detect_targets)
    sample = {
        "frames": frames,
        "move_targets": move_targets,
        "detect_targets": detect_targets,
        "legal_masks": legal_masks,
        "move_mask": move_mask,
    }
    if square_targets is not None:
        sample["square_targets"] = square_targets
    if board_corners is not None:
        sample["board_corners"] = board_corners
    if move_loss_mask is not None:
        sample["move_loss_mask"] = move_loss_mask
    if move_loss_weights is not None:
        sample["move_loss_weights"] = move_loss_weights
    return sample


def _prepare_frames(frames: torch.Tensor) -> torch.Tensor:
    """Canonicalize clip frames to float tensors in ``[0, 1]``."""
    if frames.dtype == torch.uint8:
        return frames.to(torch.float32) / 255.0
    return frames.to(torch.float32)


def _derive_move_mask(detect_targets: torch.Tensor) -> torch.Tensor:
    """Use detection targets as the source of truth for move-frame masking."""
    return detect_targets > 0.5


def _build_square_targets(clip_data: dict[str, Any]) -> torch.Tensor | None:
    fens = clip_data.get("fens")
    if fens is None:
        return None
    board_flipped = clip_data.get("board_flipped", False)
    if isinstance(board_flipped, torch.Tensor):
        board_flipped = bool(board_flipped.item())
    targets = [fen_to_square_targets(fen, board_flipped=bool(board_flipped)) for fen in fens]
    return torch.stack(targets)


def _optional_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor | None:
    if value is None:
        return None
    tensor: torch.Tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.to(dtype=dtype)


def _pad_temporal_tensor_with_last(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    current_len = tensor.shape[0]
    if current_len >= target_len:
        return tensor
    repeat_count = target_len - current_len
    padding = tensor[-1:].expand(repeat_count, *tensor.shape[1:]).clone()
    return torch.cat([tensor, padding], dim=0)


def _pad_tensor(
    tensor: torch.Tensor,
    target_len: int,
    dim: int = 0,
    fill_value: int | float = 0,
) -> torch.Tensor:
    """Pad a tensor along a given dimension to target_len."""
    current_len = tensor.shape[dim]
    if current_len >= target_len:
        return tensor
    pad_size = list(tensor.shape)
    pad_size[dim] = target_len - current_len
    padding = torch.full(
        pad_size,
        fill_value=fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=dim)
