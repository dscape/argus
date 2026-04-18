from __future__ import annotations

from types import SimpleNamespace

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
        device: str,
        batch_size: int,
    ) -> list[torch.Tensor]:
        del device, batch_size
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
