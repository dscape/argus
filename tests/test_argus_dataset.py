from __future__ import annotations

import torch

from argus.data.dataset import ArgusDataset, ArgusInMemoryDataset
from argus.data.transforms import ValidationTransform


def _make_clip(*, frames: torch.Tensor) -> dict[str, object]:
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


def test_argus_dataset_applies_transform_after_frame_preparation(tmp_path) -> None:
    clip = _make_clip(frames=torch.full((1, 3, 2, 2), 128, dtype=torch.uint8))
    torch.save(clip, tmp_path / "clip_demo_0.pt")

    sample = ArgusDataset(tmp_path, clip_length=1, transform=ValidationTransform())[0]
    expected = (128 / 255.0 - 0.485) / 0.229

    assert sample["frames"].dtype == torch.float32
    assert torch.allclose(sample["frames"][0, 0], torch.full((2, 2), expected), atol=1e-5)


def test_argus_dataset_derives_square_targets_from_fens(tmp_path) -> None:
    clip = _make_clip(frames=torch.zeros(2, 3, 2, 2, dtype=torch.float32))
    clip["fens"] = [
        "8/8/8/8/8/8/8/K6k w - - 0 1",
        "8/8/8/8/8/8/8/K6k w - - 0 1",
    ]
    clip["board_flipped"] = False
    torch.save(clip, tmp_path / "clip_demo_0.pt")

    sample = ArgusDataset(tmp_path, clip_length=2)[0]

    assert sample["square_targets"].shape == (2, 64)
    assert sample["square_targets"][0, 56].item() == 6
    assert sample["square_targets"][0, 63].item() == 12


def test_argus_dataset_preserves_board_corners_and_move_loss_weights(tmp_path) -> None:
    clip = _make_clip(frames=torch.zeros(2, 3, 2, 2, dtype=torch.float32))
    clip["board_corners"] = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[0.1, 0.0], [1.0, 0.1], [0.9, 1.0], [0.0, 0.9]],
        ],
        dtype=torch.float32,
    )
    clip["move_loss_mask"] = torch.tensor([False, True], dtype=torch.bool)
    clip["move_loss_weights"] = torch.tensor([0.0, 1.0], dtype=torch.float32)
    torch.save(clip, tmp_path / "clip_demo_0.pt")

    sample = ArgusDataset(tmp_path, clip_length=3)[0]

    assert sample["board_corners"].shape == (3, 4, 2)
    assert torch.allclose(sample["board_corners"][-1], sample["board_corners"][-2])
    assert torch.equal(sample["move_loss_mask"], torch.tensor([False, True, False]))
    assert torch.allclose(sample["move_loss_weights"], torch.tensor([0.0, 1.0, 0.0]))
