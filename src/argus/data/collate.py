"""Custom collation functions for Argus DataLoaders."""

from __future__ import annotations

import torch


def argus_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate a batch of Argus clip samples into batched tensors.

    Each sample is a dict with keys:
        frames: (T, C, H, W)
        move_targets: (T,)
        detect_targets: (T,)
        legal_masks: (T, VOCAB_SIZE)
        move_mask: (T,)

    Returns:
        Batched dict with each tensor having an added batch dimension (B, ...).
    """
    frames = torch.stack([s["frames"] for s in batch])
    move_targets = torch.stack([s["move_targets"] for s in batch])
    detect_targets = torch.stack([s["detect_targets"] for s in batch])
    legal_masks = torch.stack([s["legal_masks"] for s in batch])
    move_mask = torch.stack([s["move_mask"] for s in batch])

    return {
        "frames": frames,
        "move_targets": move_targets,
        "detect_targets": detect_targets,
        "legal_masks": legal_masks,
        "move_mask": move_mask,
    }


def multi_board_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate for multi-board scenarios with variable board counts.

    Each sample may have:
        frames: (T, C, H, W) - full scene frames
        board_crops: (T, N, C, H, W) - per-board crops
        move_targets: (T, N)
        detect_targets: (T, N)
        legal_masks: (T, N, VOCAB_SIZE)
        move_mask: (T, N)
        board_ids: (T, N)

    Pads the board dimension N to the maximum across the batch.
    """
    max_boards = max(s["board_crops"].shape[1] for s in batch)
    T = batch[0]["frames"].shape[0]
    C = batch[0]["frames"].shape[1]
    H = batch[0]["frames"].shape[2]
    W = batch[0]["frames"].shape[3]
    vocab_size = batch[0]["legal_masks"].shape[-1]
    B = len(batch)

    frames = torch.stack([s["frames"] for s in batch])

    board_crops = torch.zeros(B, T, max_boards, C, H, W, dtype=batch[0]["board_crops"].dtype)
    move_targets = torch.zeros(B, T, max_boards, dtype=batch[0]["move_targets"].dtype)
    detect_targets = torch.zeros(B, T, max_boards, dtype=batch[0]["detect_targets"].dtype)
    legal_masks = torch.zeros(B, T, max_boards, vocab_size, dtype=batch[0]["legal_masks"].dtype)
    move_mask = torch.zeros(B, T, max_boards, dtype=batch[0]["move_mask"].dtype)
    board_ids = torch.full((B, T, max_boards), -1, dtype=torch.long)

    for i, s in enumerate(batch):
        n = s["board_crops"].shape[1]
        board_crops[i, :, :n] = s["board_crops"]
        move_targets[i, :, :n] = s["move_targets"]
        detect_targets[i, :, :n] = s["detect_targets"]
        legal_masks[i, :, :n] = s["legal_masks"]
        move_mask[i, :, :n] = s["move_mask"]
        if "board_ids" in s:
            board_ids[i, :, :n] = s["board_ids"]

    return {
        "frames": frames,
        "board_crops": board_crops,
        "move_targets": move_targets,
        "detect_targets": detect_targets,
        "legal_masks": legal_masks,
        "move_mask": move_mask,
        "board_ids": board_ids,
    }
