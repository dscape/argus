from __future__ import annotations

from pathlib import Path

import pytest
import scripts.train_physical_board_probe as train_physical_board_probe
import torch
from pipeline.physical.real_board_data import PhysicalRealBoardRow
from scripts.train_physical_board_probe import (
    build_synthetic_dataset,
    resolve_selection_metric,
    sample_real_rows,
    select_real_val_source_video_ids,
    split_real_rows_by_source_video_ids,
)


def _row(source_video_id: str, frame_index: int) -> PhysicalRealBoardRow:
    return PhysicalRealBoardRow(
        clip_path=f"data/argus/train_real/{source_video_id}.pt",
        frame_index=frame_index,
        source_video_id=source_video_id,
        source_channel_handle="@demo",
        corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        labels=tuple([0] * 64),
    )


def test_resolve_selection_metric_defaults_to_real_priority_when_real_selection_exists() -> None:
    assert resolve_selection_metric("auto", has_real_selection=False) == "accuracy"
    assert resolve_selection_metric("auto", has_real_selection=True) == "non_empty_plus_macro"
    assert resolve_selection_metric("macro_f1", has_real_selection=True) == "macro_f1"


def test_select_real_val_source_video_ids_prefers_requested_ids() -> None:
    rows = [_row("video-a", 0), _row("video-b", 1), _row("video-c", 2)]

    selected = select_real_val_source_video_ids(
        rows,
        requested_source_video_ids=["video-c", "video-a"],
        requested_count=0,
        seed=7,
    )

    assert selected == ["video-a", "video-c"]


def test_split_real_rows_by_source_video_ids_holds_out_selected_sources() -> None:
    rows = [_row("video-a", 0), _row("video-b", 1), _row("video-a", 2)]

    train_rows, selection_rows = split_real_rows_by_source_video_ids(
        rows,
        selection_source_video_ids=["video-a"],
    )

    assert [row.frame_index for row in train_rows] == [1]
    assert [row.frame_index for row in selection_rows] == [0, 2]


def test_sample_real_rows_respects_max_frames() -> None:
    rows = [_row("video-a", index) for index in range(5)]

    sampled_rows = sample_real_rows(rows, max_frames=3, seed=7)

    assert len(sampled_rows) == 3
    assert {row.frame_index for row in sampled_rows}.issubset({0, 1, 2, 3, 4})


def test_build_synthetic_dataset_routes_clip_frames_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class StubClipDataset:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            assert index == 0
            return torch.zeros((3, 64, 64)), torch.zeros((64,), dtype=torch.long)

    monkeypatch.setattr(
        train_physical_board_probe,
        "PhysicalSyntheticClipBoardDataset",
        StubClipDataset,
    )

    dataset = build_synthetic_dataset(
        synthetic_clips_dir=Path("data/argus/train"),
        board_input_mode="oblique_board",
        num_positions=1,
        image_size=64,
        seed=7,
    )

    image, labels, corners = dataset[0]
    assert captured_kwargs == {
        "clips_dir": Path("data/argus/train"),
        "num_positions": 1,
        "image_size": 64,
        "seed": 7,
    }
    assert image.shape == (3, 64, 64)
    assert labels.shape == (64,)
    assert corners.shape == (4, 2)


def test_build_synthetic_dataset_supports_oblique_board_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubClipDataset:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            assert index == 0
            return torch.zeros((3, 64, 64)), torch.zeros((64,), dtype=torch.long)

    monkeypatch.setattr(
        train_physical_board_probe,
        "PhysicalSyntheticClipBoardDataset",
        StubClipDataset,
    )

    dataset = build_synthetic_dataset(
        synthetic_clips_dir=Path("data/argus/train"),
        board_input_mode="oblique_board_crop",
        num_positions=1,
        image_size=64,
        seed=7,
    )

    image, labels = dataset[0]
    assert image.shape == (3, 64, 64)
    assert labels.shape == (64,)
