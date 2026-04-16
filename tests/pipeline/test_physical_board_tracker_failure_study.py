from __future__ import annotations

import json
from types import SimpleNamespace

import chess
import numpy as np
import torch
from pipeline.physical.board_data import PhysicalEvalBoardRow
from pipeline.physical.board_tracker_failure_study import (
    TrackerFailureStudyConfig,
    create_tracker_failure_study,
)
from pipeline.shared import board_to_class_ids


def _row(
    annotation_id: str,
    *,
    clip_path: str,
    frame_index: int,
    board: chess.Board,
) -> PhysicalEvalBoardRow:
    return PhysicalEvalBoardRow(
        annotation_id=annotation_id,
        board_path=f"data/physical/val/boards/{annotation_id}.png",
        labels=tuple(board_to_class_ids(board)),
        source_video_id="video-1",
        clip_path=clip_path,
        frame_index=frame_index,
    )


def _logits_for_board(board: chess.Board, *, preferred_logit: float = 5.0) -> torch.Tensor:
    logits = torch.zeros((64, 13), dtype=torch.float32)
    for square_index, class_id in enumerate(board_to_class_ids(board)):
        logits[square_index, class_id] = preferred_logit
    return logits


def test_create_tracker_failure_study_writes_review_bundle(monkeypatch, tmp_path) -> None:
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    rows = [
        _row("frame0", clip_path="clip_a.pt", frame_index=0, board=board),
        _row("frame1", clip_path="clip_a.pt", frame_index=1, board=moved_board),
        _row("frame2", clip_path="clip_a.pt", frame_index=2, board=moved_board),
    ]
    logits_by_frame = {
        0: _logits_for_board(board),
        1: _logits_for_board(board),
        2: _logits_for_board(board),
    }

    import pipeline.physical.board_tracker_failure_study as failure_study

    monkeypatch.setattr(
        failure_study,
        "PhysicalEvalBoardDataset",
        lambda: SimpleNamespace(rows=rows),
    )
    monkeypatch.setattr(
        failure_study,
        "_initial_board_state",
        lambda clip_path, clip_state_cache: (chess.STARTING_BOARD_FEN, "w"),
    )
    monkeypatch.setattr(
        failure_study,
        "_load_row_image",
        lambda row, observation_input, clip_cache: np.full(
            (8, 8, 3),
            int(row.frame_index or 0),
            dtype=np.uint8,
        ),
    )

    def fake_read_board_logits_batch_from_frames(
        images: list[np.ndarray],
        *,
        corners_list: list[object] | None,
        device: str,
        weights_path: str | None,
        batch_size: int,
    ) -> list[torch.Tensor]:
        del corners_list, device, weights_path, batch_size
        return [logits_by_frame[int(image[0, 0, 0])].clone() for image in images]

    monkeypatch.setattr(
        failure_study,
        "read_board_logits_batch_from_frames",
        fake_read_board_logits_batch_from_frames,
    )

    summary = create_tracker_failure_study(
        config=TrackerFailureStudyConfig(
            observation_input="rectified_board",
            temporal_mode="off",
            tracker_mode="lookahead",
            lookahead_window=3,
            lookahead_margin=8.0,
        ),
        output_dir=tmp_path,
        limit=10,
        panel_size=64,
        top_legal_candidates=3,
        sample_mode="first",
    )

    assert summary["total_failures"] == 2
    assert summary["selected_failures"] == 2
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "manual_buckets.csv").exists()
    assert (tmp_path / "BUCKETS.md").exists()
    assert (tmp_path / "contact_sheet.png").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [item["annotation_id"] for item in manifest] == ["frame1", "frame2"]
    assert manifest[0]["suggested_root_cause"] == "tracker_boundary_jitter"
    assert manifest[1]["suggested_root_cause"] == "rectification_or_classifier"
    assert manifest[1]["legal_from_previous_decoded"]["gt_is_legal_successor"] is True
    assert manifest[1]["legal_from_previous_decoded"]["best_legal_matches_gt"] is False
    assert manifest[1]["image_path"].endswith("_f0002.png")
