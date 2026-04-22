from __future__ import annotations

from types import SimpleNamespace

import chess
import numpy as np
import torch
from pipeline.physical.board_probe import runtime_visualization
from pipeline.physical.board_probe.board_data import PhysicalEvalBoardRow


def _row(frame_index: int) -> PhysicalEvalBoardRow:
    return PhysicalEvalBoardRow(
        annotation_id=f"clip_frame_{frame_index:04d}",
        board_path=f"data/physical/val/boards/frame_{frame_index:04d}.png",
        labels=(frame_index,) * 64,
        source_video_id="video-1",
        corners=((0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)),
        clip_path="data/argus/train_real/example_clip.pt",
        frame_index=frame_index,
    )


def test_collect_visualized_frames_only_processes_history_up_to_selected_frame(
    monkeypatch,
) -> None:
    processed_batches: list[list[int]] = []

    monkeypatch.setattr(
        runtime_visualization,
        "_load_board_image",
        lambda row: np.full((4, 4, 3), int(row.frame_index or 0), dtype=np.uint8),
    )

    def fake_read_board_logits_batch_from_frames(
        images: list[np.ndarray],
        *,
        corners_list: list[object] | None,
        device: str,
        batch_size: int,
    ) -> list[torch.Tensor]:
        del device, batch_size
        assert corners_list is not None
        frame_indices = [int(image[0, 0, 0]) for image in images]
        processed_batches.append(frame_indices)
        return [torch.full((64, 13), float(frame_index)) for frame_index in frame_indices]

    monkeypatch.setattr(
        runtime_visualization,
        "read_board_logits_batch_from_frames",
        fake_read_board_logits_batch_from_frames,
    )

    class FakeSequenceReader:
        def __init__(self, *, device: str) -> None:
            del device

        def smooth_logits(self, logits: torch.Tensor) -> torch.Tensor:
            return logits + 100.0

    monkeypatch.setattr(
        runtime_visualization,
        "PhysicalBoardLogitsSequenceReader",
        FakeSequenceReader,
    )
    monkeypatch.setattr(
        runtime_visualization,
        "board_observation_from_logits",
        lambda logits, timestamp_seconds=0.0: SimpleNamespace(
            fen=f"f{int(logits[0, 0].item())}",
            square_confidences=(0.5,) * 64,
            timestamp_seconds=timestamp_seconds,
        ),
    )
    monkeypatch.setattr(
        runtime_visualization,
        "_fen_to_class_ids",
        lambda fen: [int(fen[1:])] * 64,
    )

    frames = runtime_visualization._collect_visualized_frames_for_indices(
        [_row(frame_index) for frame_index in range(4)],
        selected_frame_indices={2},
        device="cpu",
    )

    assert processed_batches == [[0, 1, 2]]
    assert [frame.frame_index for frame in frames] == [2]
    assert frames[0].gt_change_count == 64
    assert frames[0].stateless_change_count == 64
    assert frames[0].temporal_change_count == 64
    assert frames[0].stateless_error_count == 0
    assert frames[0].temporal_error_count == 64


def test_collect_visualized_frames_exposes_legal_decoder_transition_labels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_visualization,
        "_load_board_image",
        lambda row: np.full((4, 4, 3), int(row.frame_index or 0), dtype=np.uint8),
    )
    monkeypatch.setattr(
        runtime_visualization,
        "read_board_logits_batch_from_frames",
        lambda images, **kwargs: [torch.zeros((64, 13), dtype=torch.float32) for _ in images],
    )

    class FakeSequenceReader:
        def __init__(self, *, device: str) -> None:
            del device

        def smooth_logits(self, logits: torch.Tensor) -> torch.Tensor:
            return logits

    monkeypatch.setattr(
        runtime_visualization,
        "PhysicalBoardLogitsSequenceReader",
        FakeSequenceReader,
    )
    monkeypatch.setattr(
        runtime_visualization,
        "board_observation_from_logits",
        lambda logits, timestamp_seconds=0.0: SimpleNamespace(
            fen="4k3/8/8/8/8/8/8/4K3",
            square_confidences=(0.5,) * 64,
            timestamp_seconds=timestamp_seconds,
        ),
    )
    monkeypatch.setattr(
        runtime_visualization,
        "_load_initial_board_state",
        lambda clip_path: (chess.STARTING_BOARD_FEN, "w"),
    )
    monkeypatch.setattr(
        runtime_visualization,
        "decode_sequence_with_production_decoder",
        lambda logits, *, initial_board_fen, initial_side_to_move: [
            SimpleNamespace(
                fen="4k3/8/8/8/8/8/8/4K3",
                full_fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                move_uci="e2e4",
            )
        ],
    )

    frames = runtime_visualization._collect_visualized_frames_for_indices(
        [_row(0)],
        selected_frame_indices={0},
        device="cpu",
    )

    assert frames[0].temporal_transition_kind == "legal_move"
    assert frames[0].temporal_transition_label == "e2e4"


def test_classify_transition_marks_illegal_single_piece_transfer_promotion() -> None:
    previous_full_fen = "k7/4P3/8/8/8/8/8/7K w - - 0 1"
    current_class_ids = tuple(runtime_visualization._fen_to_class_ids("k4Q2/8/8/8/8/8/8/7K"))

    transition = runtime_visualization._classify_transition(previous_full_fen, current_class_ids)

    assert transition.kind == "illegal_single_piece_transfer"
    assert transition.label == "e7->f8=Q"
