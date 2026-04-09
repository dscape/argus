from __future__ import annotations

import torch

from argus.data.dataset import ArgusDataset, ArgusInMemoryDataset


def _make_clip(*, frames: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "frames": frames,
        "move_targets": torch.tensor([0, 1, 0], dtype=torch.long),
        "detect_targets": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        "legal_masks": torch.zeros(3, 4, dtype=torch.bool),
        # Legacy real clips wrote an all-ones float mask; dataset loading should not trust it.
        "move_mask": torch.ones(3, dtype=torch.float32),
    }


def test_argus_dataset_normalizes_uint8_frames_and_derives_move_mask(tmp_path) -> None:
    clip = _make_clip(frames=torch.full((3, 3, 2, 2), 128, dtype=torch.uint8))
    torch.save(clip, tmp_path / "clip_demo_0.pt")

    sample = ArgusDataset(tmp_path, clip_length=5)[0]

    assert sample["frames"].dtype == torch.float32
    assert torch.allclose(sample["frames"][0], torch.full((3, 2, 2), 128 / 255.0))
    assert torch.equal(
        sample["move_mask"],
        torch.tensor([False, True, False, False, False], dtype=torch.bool),
    )


def test_argus_in_memory_dataset_preserves_float_frames_and_derives_move_mask() -> None:
    frames = torch.rand(3, 3, 2, 2, dtype=torch.float32)
    sample = ArgusInMemoryDataset([_make_clip(frames=frames)], clip_length=3)[0]

    assert sample["frames"].dtype == torch.float32
    assert torch.allclose(sample["frames"], frames)
    assert torch.equal(sample["move_mask"], torch.tensor([False, True, False], dtype=torch.bool))
