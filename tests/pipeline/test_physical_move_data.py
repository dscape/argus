from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
from pipeline.physical.shared.move_data import (
    _build_real_clip_sample,
    _build_replay_supervision_targets,
    _infer_annotated_moves,
    _replay_targets_for_clip,
    _slice_move_windows,
    build_board_hypotheses_from_piece_fen,
    load_eval_move_sequences,
    load_real_move_sequences,
)
from pipeline.shared import board_to_class_ids

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary


@dataclass(frozen=True)
class _Row:
    labels: tuple[int, ...]


@dataclass(frozen=True)
class _RealClipRow:
    corners: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _RealSequenceRow:
    clip_path: str
    frame_index: int
    source_video_id: str | None
    corners: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]


def _write_test_video(path: Path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def test_replay_targets_for_clip_builds_move_and_detect_targets() -> None:
    vocab = get_vocabulary()
    clip = {
        "frames": torch.zeros((3, 3, 8, 8), dtype=torch.float32),
        "initial_board_fen": chess.STARTING_BOARD_FEN,
        "move_ucis": ["e2e4"],
        "move_frame_indices": [1],
        "frame_indices": [0, 1, 2],
    }

    replay_targets = _replay_targets_for_clip(clip)

    assert replay_targets is not None
    move_targets, detect_targets, legal_masks, board_fens = replay_targets
    assert move_targets.tolist() == [NO_MOVE_IDX, vocab.uci_to_index("e2e4"), NO_MOVE_IDX]
    assert detect_targets.tolist() == [0.0, 1.0, 0.0]
    assert legal_masks.shape[0] == 3
    assert board_fens[0].startswith(chess.STARTING_BOARD_FEN)


def test_build_replay_supervision_targets_supports_causal_tolerance() -> None:
    vocab = get_vocabulary()
    clip = {
        "frames": torch.zeros((5, 3, 8, 8), dtype=torch.float32),
        "initial_board_fen": chess.STARTING_BOARD_FEN,
        "move_ucis": ["e2e4"],
        "move_frame_indices": [3],
        "frame_indices": [0, 1, 2, 3, 4],
    }

    replay_targets = _build_replay_supervision_targets(
        clip,
        move_target_pre_frames=2,
        detect_target_radius=1,
        detect_target_decay=0.5,
    )

    assert replay_targets is not None
    move_targets, detect_targets, _legal_masks, _board_fens, move_loss_mask, move_loss_weights = (
        replay_targets
    )
    expected_move = vocab.uci_to_index("e2e4")
    assert move_targets.tolist() == [
        NO_MOVE_IDX,
        expected_move,
        expected_move,
        expected_move,
        NO_MOVE_IDX,
    ]
    assert detect_targets.tolist() == [0.0, 0.0, 0.5, 1.0, 0.5]
    assert move_loss_mask.tolist() == [False, True, True, True, False]
    assert move_loss_weights.tolist() == [0.0, 0.25, 0.5, 1.0, 0.0]


def test_build_board_hypotheses_from_piece_fen_uses_known_side_to_move() -> None:
    boards = build_board_hypotheses_from_piece_fen(
        chess.STARTING_BOARD_FEN, initial_side_to_move="b"
    )

    assert len(boards) == 1
    assert boards[0].turn is chess.BLACK


def test_infer_annotated_moves_recovers_single_legal_transition() -> None:
    vocab = get_vocabulary()
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")
    rows = [
        _Row(labels=tuple(board_to_class_ids(board))),
        _Row(labels=tuple(board_to_class_ids(moved_board))),
    ]

    move_targets, detect_targets = _infer_annotated_moves(rows, chess.STARTING_BOARD_FEN)

    assert move_targets == [NO_MOVE_IDX, vocab.uci_to_index("e2e4")]
    assert detect_targets == [0.0, 1.0]


def test_load_eval_move_sequences_supports_piece_projection_board(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = chess.Board()
    annotation_root = tmp_path / "physical_val"
    annotation_root.mkdir(parents=True)
    labels = board_to_class_ids(board)
    (annotation_root / "board_annotations.jsonl").write_text(
        json.dumps(
            {
                "annotation_id": "clip_frame0000",
                "clip_path": "unused_clip.pt",
                "frame_index": 0,
                "source_video_id": "video123",
                "corners": [[0, 0], [19, 0], [19, 19], [0, 19]],
                "labels": labels,
                "corner_space": "clip_frame",
                "clip_frame_size": [20, 20],
                "native_corners": [[0, 0], [19, 0], [19, 19], [0, 19]],
                "native_image_bbox": [0, 0, 20, 20],
                "source_frame_index": 0,
            }
        )
        + "\n"
    )

    clip_path = tmp_path / "unused_clip.pt"
    torch.save(
        {"frames": torch.full((1, 3, 20, 20), 180, dtype=torch.uint8)},
        clip_path,
    )

    import pipeline.physical.shared.annotation_rows as annotation_rows
    import pipeline.physical.shared.move_data as move_data

    monkeypatch.setattr(move_data, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(annotation_rows, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        move_data,
        "_initial_board_state_for_clip",
        lambda clip_path: (chess.STARTING_BOARD_FEN, "w"),
    )

    sequences = load_eval_move_sequences(
        annotation_root=annotation_root,
        image_size=16,
        observation_mode="piece_projection_board",
        board_crop_margin=0.0,
    )

    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.frames.shape == (1, 3, 16, 16)
    assert sequence.board_corners is not None
    assert sequence.board_corners.shape == (1, 4, 2)
    assert torch.all(sequence.board_corners >= 0)
    assert torch.all(sequence.board_corners <= 16)


def test_slice_move_windows_can_repeat_positive_windows() -> None:
    clip_data = {
        "frames": torch.zeros((8, 3, 8, 8), dtype=torch.float32),
        "move_targets": torch.tensor([NO_MOVE_IDX] * 8, dtype=torch.long),
        "detect_targets": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        "legal_masks": torch.zeros((8, 2), dtype=torch.bool),
        "exact_move_mask": torch.tensor([False, False, False, False, False, True, False, False]),
    }

    windows = _slice_move_windows(
        clip_data,
        clip_length=4,
        negative_window_stride=2,
        max_negative_windows=1,
        positive_window_repeat=3,
    )

    positive_windows = [window for window in windows if window[3]]
    negative_windows = [window for window in windows if not window[3]]
    assert len(positive_windows) == 3
    assert len(negative_windows) == 1
    assert {window[1] for window in positive_windows} == {3}


def test_load_real_move_sequences_supports_internal_selection_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    selected_clip_path = "clip_selected.pt"
    other_clip_path = "clip_other.pt"
    clip_payload = {
        "frames": torch.zeros((2, 3, 8, 8), dtype=torch.uint8),
        "initial_board_fen": chess.STARTING_BOARD_FEN,
        "initial_side_to_move": "w",
        "move_ucis": ["e2e4"],
        "move_frame_indices": [1],
        "frame_indices": [0, 1],
    }
    torch.save(clip_payload, tmp_path / selected_clip_path)
    torch.save(
        {**clip_payload, "move_ucis": [], "move_frame_indices": []},
        tmp_path / other_clip_path,
    )

    import pipeline.physical.shared.move_data as move_data

    monkeypatch.setattr(move_data, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        move_data,
        "load_real_board_rows",
        lambda **_kwargs: [
            _RealSequenceRow(
                clip_path=selected_clip_path,
                frame_index=0,
                source_video_id="selected",
                corners=((0.0, 0.0), (7.0, 0.0), (7.0, 7.0), (0.0, 7.0)),
                labels=tuple(board_to_class_ids(board)),
            ),
            _RealSequenceRow(
                clip_path=selected_clip_path,
                frame_index=1,
                source_video_id="selected",
                corners=((0.0, 0.0), (7.0, 0.0), (7.0, 7.0), (0.0, 7.0)),
                labels=tuple(board_to_class_ids(moved_board)),
            ),
            _RealSequenceRow(
                clip_path=other_clip_path,
                frame_index=0,
                source_video_id="other",
                corners=((0.0, 0.0), (7.0, 0.0), (7.0, 7.0), (0.0, 7.0)),
                labels=tuple(board_to_class_ids(board)),
            ),
            _RealSequenceRow(
                clip_path=other_clip_path,
                frame_index=1,
                source_video_id="other",
                corners=((0.0, 0.0), (7.0, 0.0), (7.0, 7.0), (0.0, 7.0)),
                labels=tuple(board_to_class_ids(board)),
            ),
        ],
    )

    sequences = load_real_move_sequences(
        clips_dir=tmp_path,
        image_size=8,
        selection_source_video_ids={"selected"},
        observation_mode="piece_projection_board",
        board_crop_margin=0.0,
    )

    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.clip_path == selected_clip_path
    assert sequence.source_video_id == "selected"
    assert sequence.frames.shape == (2, 3, 8, 8)
    assert sequence.board_corners is not None
    assert sequence.labels == (
        tuple(board_to_class_ids(board)),
        tuple(board_to_class_ids(moved_board)),
    )
    assert sequence.inferred_detect_targets == (0.0, 1.0)
    assert sequence.inferred_move_targets[0] == NO_MOVE_IDX
    assert sequence.inferred_move_targets[1] == get_vocabulary().uci_to_index("e2e4")


def test_build_real_clip_sample_supports_piece_projection_board(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clip = {
        "frames": torch.stack(
            [
                torch.from_numpy(np.full((20, 20, 3), 40, dtype=np.uint8)).permute(2, 0, 1),
                torch.from_numpy(np.full((20, 20, 3), 80, dtype=np.uint8)).permute(2, 0, 1),
            ],
            dim=0,
        ),
        "initial_board_fen": chess.STARTING_BOARD_FEN,
        "move_ucis": [],
        "move_frame_indices": [],
        "frame_indices": [0, 1],
    }

    clip_data = _build_real_clip_sample(
        clip,
        clip_path="data/argus/train_real/clip_overlay_video123_clip1_0.pt",
        clip_rows=[_RealClipRow(corners=((0.0, 0.0), (19.0, 0.0), (19.0, 19.0), (0.0, 19.0)))],
        image_size=16,
        observation_mode="piece_projection_board",
        move_target_pre_frames=0,
        detect_target_radius=0,
        detect_target_decay=0.5,
        board_crop_margin=0.0,
    )

    assert clip_data is not None
    assert clip_data["frames"].shape == (2, 3, 16, 16)
    assert "board_corners" in clip_data
    assert clip_data["board_corners"].shape == (2, 4, 2)
