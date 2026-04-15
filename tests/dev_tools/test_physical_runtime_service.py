from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from api.services.evaluate import physical_runtime_service
from PIL import Image
from pipeline.physical.board_data import PhysicalEvalBoardRow
from pipeline.physical.runtime_visualization import VisualizedRuntimeFrame


def _row(annotation_id: str, clip_path: str, frame_index: int) -> PhysicalEvalBoardRow:
    return PhysicalEvalBoardRow(
        annotation_id=annotation_id,
        board_path=f"data/physical/val/boards/{annotation_id}.png",
        labels=(frame_index % 13,) * 64,
        source_video_id="video-1",
        clip_path=clip_path,
        frame_index=frame_index,
    )


def _visualized_frame(row: PhysicalEvalBoardRow) -> VisualizedRuntimeFrame:
    labels = tuple(int(value) for value in row.labels)
    confidences = (0.5,) * 64
    no_changes = (False,) * 64
    return VisualizedRuntimeFrame(
        frame_index=int(row.frame_index or 0),
        annotation_id=row.annotation_id,
        board_path=row.board_path,
        crop_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        gt_class_ids=labels,
        stateless_class_ids=labels,
        temporal_class_ids=labels,
        stateless_confidences=confidences,
        temporal_confidences=confidences,
        stateless_error_count=0,
        temporal_error_count=0,
        gt_change_count=0,
        stateless_change_count=0,
        temporal_change_count=0,
        gt_changed_mask=no_changes,
        stateless_changed_mask=no_changes,
        temporal_changed_mask=no_changes,
    )


def test_inspect_runtime_frames_groups_by_clip_and_preserves_input_order(monkeypatch) -> None:
    rows = [
        _row("clip-a-frame-0001", "data/argus/train_real/clip_a.pt", 1),
        _row("clip-a-frame-0002", "data/argus/train_real/clip_a.pt", 2),
        _row("clip-b-frame-0001", "data/argus/train_real/clip_b.pt", 1),
    ]
    monkeypatch.setattr(
        physical_runtime_service,
        "PhysicalEvalBoardDataset",
        lambda: SimpleNamespace(rows=rows),
    )

    clip_calls: list[tuple[str, set[int]]] = []

    def fake_collect_visualized_frames_for_indices(
        clip_rows: list[PhysicalEvalBoardRow],
        *,
        selected_frame_indices: set[int],
        device: str,
    ) -> list[VisualizedRuntimeFrame]:
        del device
        clip_calls.append((str(clip_rows[0].clip_path), selected_frame_indices))
        return [
            _visualized_frame(row) for row in clip_rows if row.frame_index in selected_frame_indices
        ]

    monkeypatch.setattr(
        physical_runtime_service,
        "_collect_visualized_frames_for_indices",
        fake_collect_visualized_frames_for_indices,
    )
    monkeypatch.setattr(
        physical_runtime_service,
        "render_visualized_runtime_frame",
        lambda frame, panel_size: Image.new("RGB", (panel_size, panel_size)),
    )

    results = physical_runtime_service.inspect_runtime_frames(
        annotation_ids=["clip-b-frame-0001", "clip-a-frame-0002", "clip-a-frame-0001"],
        panel_size=32,
        device="cpu",
    )

    assert [result["annotation_id"] for result in results] == [
        "clip-b-frame-0001",
        "clip-a-frame-0002",
        "clip-a-frame-0001",
    ]
    assert clip_calls == [
        ("data/argus/train_real/clip_b.pt", {1}),
        ("data/argus/train_real/clip_a.pt", {1, 2}),
    ]
    assert all(result["temporal_exact_match"] for result in results)
