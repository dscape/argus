from __future__ import annotations

import json
from types import SimpleNamespace

import chess
import numpy as np
import torch
from pipeline.physical.board_probe.board_data import PhysicalEvalBoardRow
from pipeline.physical.board_probe.failure_study import (
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
    source_video_id: str = "video-1",
) -> PhysicalEvalBoardRow:
    return PhysicalEvalBoardRow(
        annotation_id=annotation_id,
        board_path=f"data/physical/val/boards/{annotation_id}.png",
        labels=tuple(board_to_class_ids(board)),
        source_video_id=source_video_id,
        corners=((0.0, 0.0), (7.0, 0.0), (7.0, 7.0), (0.0, 7.0)),
        clip_path=clip_path,
        frame_index=frame_index,
    )


def _logits_for_board(board: chess.Board, *, preferred_logit: float = 5.0) -> torch.Tensor:
    logits = torch.zeros((64, 13), dtype=torch.float32)
    for square_index, class_id in enumerate(board_to_class_ids(board)):
        logits[square_index, class_id] = preferred_logit
    return logits


def test_create_tracker_failure_study_writes_episode_bundle(monkeypatch, tmp_path) -> None:
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

    import pipeline.physical.board_probe.failure_study as failure_study

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
            preceding_frames=1,
            recovery_gap=1,
            max_per_video=5,
        ),
        output_dir=tmp_path,
        limit=10,
        panel_size=64,
        top_legal_candidates=3,
    )

    # frames 1 and 2 fail together (no recovery between them) → 1 episode
    assert summary["total_episodes"] == 1
    assert summary["selected_episodes"] == 1
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "manual_buckets.csv").exists()
    assert (tmp_path / "BUCKETS.md").exists()
    assert (tmp_path / "contact_sheet.png").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest) == 1
    episode = manifest[0]
    assert episode["episode_id"] == "ep001"
    assert episode["source_video_id"] == "video-1"
    assert episode["first_frame_index"] == 1
    assert episode["last_frame_index"] == 2
    assert episode["length"] == 2

    failing = episode["failing_frame"]
    assert failing["annotation_id"] == "frame1"
    assert failing["is_failing_frame"] is True
    assert failing["offset_from_failure"] == 0
    assert failing["image_path"].endswith("frame_0001.png")

    # One preceding frame (frame0) because preceding_frames=1
    assert len(episode["preceding_frames"]) == 1
    preceding = episode["preceding_frames"][0]
    assert preceding["annotation_id"] == "frame0"
    assert preceding["is_failing_frame"] is False
    assert preceding["offset_from_failure"] == -1
    assert preceding["image_path"].endswith("frame_0000.png")

    assert episode["suggested_bucket"] == "temporal in-between / move execution ambiguity"


def test_max_per_video_caps_selection(monkeypatch, tmp_path) -> None:
    board = chess.Board()
    moved_board = board.copy(stack=False)
    moved_board.push_uci("e2e4")

    # Two clips from the same video, each producing one failure episode.
    rows: list[PhysicalEvalBoardRow] = []
    for clip_index in range(2):
        rows.append(
            _row(
                f"clip{clip_index}_f0",
                clip_path=f"clip_{clip_index}.pt",
                frame_index=0,
                board=board,
                source_video_id="video-A",
            )
        )
        rows.append(
            _row(
                f"clip{clip_index}_f1",
                clip_path=f"clip_{clip_index}.pt",
                frame_index=1,
                board=moved_board,
                source_video_id="video-A",
            )
        )

    # All frames decode to the starting board, so each "frame1" is a failure.
    logits_by_annotation = {row.annotation_id: _logits_for_board(board) for row in rows}

    import pipeline.physical.board_probe.failure_study as failure_study

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

    def fake_load_row_image(row, observation_input, clip_cache):
        # Encode annotation_id into the image so we can route to the right logits.
        # Use a small hash that fits in uint8 channels.
        token = abs(hash(row.annotation_id)) % 251
        return np.full((8, 8, 3), token, dtype=np.uint8)

    monkeypatch.setattr(failure_study, "_load_row_image", fake_load_row_image)

    token_to_annotation = {abs(hash(row.annotation_id)) % 251: row.annotation_id for row in rows}

    def fake_read_board_logits_batch_from_frames(
        images, *, corners_list, device, weights_path, batch_size
    ):
        del corners_list, device, weights_path, batch_size
        return [
            logits_by_annotation[token_to_annotation[int(image[0, 0, 0])]].clone()
            for image in images
        ]

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
            preceding_frames=0,
            recovery_gap=1,
            max_per_video=1,
        ),
        output_dir=tmp_path,
        limit=10,
        panel_size=64,
        top_legal_candidates=3,
    )

    # Two episodes exist (one per clip) but max_per_video=1 caps selection to 1.
    assert summary["total_episodes"] == 2
    assert summary["selected_episodes"] == 1
    assert summary["selected_per_video_counts"] == {"video-A": 1}
