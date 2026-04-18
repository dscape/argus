from __future__ import annotations

import json

import chess
import numpy as np
import torch
from pipeline.physical.shared.real_board_data import (
    PhysicalRealBoardDataset,
    PhysicalRealBoardRow,
    build_excluded_move_neighborhood,
    infer_channel_corner_templates,
    replay_clip_display_fens,
    replay_clip_move_sample_indices,
)


def test_replay_clip_display_fens_uses_post_move_board_on_move_frame() -> None:
    clip = {
        "frames": torch.zeros((2, 3, 8, 8), dtype=torch.uint8),
        "initial_board_fen": chess.STARTING_BOARD_FEN,
        "move_ucis": ["e2e4"],
        "move_frame_indices": torch.tensor([1], dtype=torch.long),
    }

    fens = replay_clip_display_fens(clip)

    board = chess.Board()
    assert fens[0] == board.fen()
    board.push(chess.Move.from_uci("e2e4"))
    assert fens[1] == board.fen()


def test_replay_clip_display_fens_maps_absolute_move_frames_to_sample_indices() -> None:
    clip = {
        "frames": torch.zeros((4, 3, 8, 8), dtype=torch.uint8),
        "frame_indices": torch.tensor([210, 225, 240, 255], dtype=torch.long),
        "initial_board_fen": chess.STARTING_BOARD_FEN,
        "move_ucis": ["e2e4"],
        "move_frame_indices": torch.tensor([240], dtype=torch.long),
    }

    fens = replay_clip_display_fens(clip)

    board = chess.Board()
    assert fens[0] == board.fen()
    assert fens[1] == board.fen()
    board.push(chess.Move.from_uci("e2e4"))
    assert fens[2] == board.fen()
    assert fens[3] == board.fen()


def test_replay_clip_move_sample_indices_uses_sampled_frame_indices() -> None:
    clip = {
        "frames": torch.zeros((4, 3, 8, 8), dtype=torch.uint8),
        "frame_indices": torch.tensor([210, 225, 240, 255], dtype=torch.long),
        "move_frame_indices": torch.tensor([240], dtype=torch.long),
    }

    assert replay_clip_move_sample_indices(clip) == {2}


def test_build_excluded_move_neighborhood_excludes_local_margin() -> None:
    assert build_excluded_move_neighborhood({2}, total_frames=5, neighborhood=1) == {1, 2, 3}
    assert build_excluded_move_neighborhood({2}, total_frames=5, neighborhood=-1) == set()


def test_infer_channel_corner_templates_reads_eval_annotations(tmp_path) -> None:
    project_root = tmp_path
    clips_dir = project_root / "data" / "argus" / "train_real"
    clips_dir.mkdir(parents=True)
    clip_path = clips_dir / "clip_overlay_demo_clip0_0.pt"
    torch.save(
        {
            "frames": torch.zeros((1, 3, 8, 8), dtype=torch.uint8),
            "source_channel_handle": "@demo",
        },
        clip_path,
    )

    eval_root = project_root / "data" / "physical" / "eval"
    eval_root.mkdir(parents=True)
    annotations_path = eval_root / "board_annotations.jsonl"
    annotations_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "clip_path": "data/argus/train_real/clip_overlay_demo_clip0_0.pt",
                        "corners": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    }
                ),
                json.dumps(
                    {
                        "clip_path": "data/argus/train_real/clip_overlay_demo_clip0_0.pt",
                        "corners": [[2, 2], [12, 2], [12, 12], [2, 12]],
                    }
                ),
            ]
        )
    )

    import pipeline.physical.shared.real_board_data as real_board_data

    original_root = real_board_data._PROJECT_ROOT
    real_board_data._PROJECT_ROOT = project_root
    try:
        templates = infer_channel_corner_templates(eval_root=eval_root)
    finally:
        real_board_data._PROJECT_ROOT = original_root

    assert templates["@demo"] == ((1.0, 1.0), (11.0, 1.0), (11.0, 11.0), (1.0, 11.0))


def test_physical_real_board_dataset_loads_board_neighborhood_frame(tmp_path) -> None:
    project_root = tmp_path
    clip_path = project_root / "data" / "argus" / "train_real" / "clip_overlay_demo_clip0_0.pt"
    clip_path.parent.mkdir(parents=True)
    frames = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)
    frames[0, :, 8:24, 8:24] = 255
    torch.save({"frames": frames}, clip_path)

    dataset = PhysicalRealBoardDataset(
        rows=[
            PhysicalRealBoardRow(
                clip_path="data/argus/train_real/clip_overlay_demo_clip0_0.pt",
                frame_index=0,
                source_video_id="demo",
                source_channel_handle="@demo",
                corners=((8.0, 8.0), (23.0, 8.0), (23.0, 23.0), (8.0, 23.0)),
                labels=tuple([0] * 64),
            )
        ],
        image_size=32,
    )

    import pipeline.physical.shared.real_board_data as real_board_data

    original_root = real_board_data._PROJECT_ROOT
    real_board_data._PROJECT_ROOT = project_root
    try:
        image, targets, corners = dataset[0]
    finally:
        real_board_data._PROJECT_ROOT = original_root

    assert image.shape == (3, 32, 32)
    assert targets.shape == (64,)
    assert corners.shape == (4, 2)
    assert np.isfinite(image.numpy()).all()
