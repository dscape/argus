"""Tests for clip inspection service replay validation, previews, and notes."""

from __future__ import annotations

import io

import chess
import cv2
import numpy as np
import torch
from api.services.data import clip_service

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary

VOCAB = get_vocabulary()
EXPECTED_INITIAL_BOARD_FEN = "rnbqkbnr/pp1p1ppp/2p1p3/8/4P3/3P2P1/PPP2P1P/RNBQKBNR"


def _make_midgame_clip_bytes() -> bytes:
    board = chess.Board()
    for uci in ["e2e4", "c7c6", "d2d3", "e7e6", "g2g3"]:
        board.push(chess.Move.from_uci(uci))

    move_targets = torch.full((4,), NO_MOVE_IDX, dtype=torch.long)
    move_targets[2] = VOCAB.uci_to_index("f1g2")

    clip = {
        "frames": torch.zeros((4, 3, 224, 224), dtype=torch.uint8),
        "move_targets": move_targets,
        "detect_targets": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        "frame_indices": torch.tensor([120, 135, 150, 165], dtype=torch.long),
        "frame_timestamps_seconds": torch.tensor([4.0, 4.5, 5.0, 5.5], dtype=torch.float32),
        "move_timestamps_seconds": torch.tensor([5.0], dtype=torch.float32),
        "estimated_otb_frame_indices": torch.tensor([135], dtype=torch.long),
        "estimated_otb_timestamps_seconds": torch.tensor([4.5], dtype=torch.float32),
        "initial_board_fen": board.board_fen(),
        "pgn_moves": "Bg2",
        "training_target_timing": "overlay_confirm_post_move",
        "estimated_otb_delay_seconds": 0.5,
    }

    buffer = io.BytesIO()
    torch.save(clip, buffer)
    return buffer.getvalue()


def test_inspect_replays_from_initial_board_fen_for_midgame_clip() -> None:
    session_id = clip_service.create_session(_make_midgame_clip_bytes(), "midgame.pt")
    try:
        result = clip_service.inspect(session_id)
    finally:
        clip_service.delete_session(session_id)

    assert result["replay_valid"] is True
    assert result["replay_error"] is None
    assert result["moves"][0]["uci"] == "f1g2"
    assert result["moves"][0]["san"] == "Bg2"
    assert result["moves"][0]["timestamp_seconds"] == 5.0
    assert result["moves"][0]["estimated_otb_frame_index"] == 135
    assert result["moves"][0]["estimated_otb_timestamp_seconds"] == 4.5
    assert result["moves"][0]["side_to_move"] == "white"
    assert result["moves"][0]["fen_before"] is not None
    assert result["moves"][0]["fen_after"] is not None
    assert result["frame_indices"] == [120, 135, 150, 165]
    assert result["frame_timestamps_seconds"] == [4.0, 4.5, 5.0, 5.5]
    assert result["frame_replay_fens"] == [
        "rnbqkbnr/pp1p1ppp/2p1p3/8/4P3/3P2P1/PPP2P1P/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pp1p1ppp/2p1p3/8/4P3/3P2P1/PPP2P1P/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pp1p1ppp/2p1p3/8/4P3/3P2P1/PPP2P1P/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pp1p1ppp/2p1p3/8/4P3/3P2P1/PPP2PBP/RNBQK1NR b KQkq - 1 1",
    ]
    assert result["metadata"]["initial_board_fen"] == EXPECTED_INITIAL_BOARD_FEN
    assert result["metadata"]["training_target_timing"] == "overlay_confirm_post_move"
    assert result["metadata"]["estimated_otb_delay_seconds"] == 0.5


def test_get_overlay_frame_png_uses_source_video_and_db_clip(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "demo123.avi"
    width = height = 64
    overlay_bbox = (10, 12, 16, 20)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (width, height),
    )
    assert writer.isOpened()

    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
    ]
    for color in colors:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x, y, w, h = overlay_bbox
        frame[y : y + h, x : x + w] = color
        writer.write(frame)
    writer.release()

    clip = {
        "frames": torch.zeros((3, 3, 224, 224), dtype=torch.uint8),
        "frame_indices": torch.tensor([0, 1, 2], dtype=torch.long),
        "source_video_id": "demo123",
    }
    buffer = io.BytesIO()
    torch.save(clip, buffer)

    monkeypatch.setattr(
        clip_service,
        "_load_db_clip_overlay_row",
        lambda clip_id: {
            "id": clip_id,
            "video_id": "demo123",
            "overlay_bbox": overlay_bbox,
            "ref_resolution": (64, 64),
        },
    )
    monkeypatch.setattr(clip_service, "_get_video_path", lambda video_id: str(video_path))

    session_id = clip_service.create_session(
        buffer.getvalue(),
        "clip_overlay_demo123_clip9_0.pt",
        source_filepath=str(video_path),
    )
    try:
        png_bytes = clip_service.get_overlay_frame_png(session_id, 1)
    finally:
        clip_service.delete_session(session_id)

    decoded = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (overlay_bbox[3], overlay_bbox[2])
    assert float(decoded[:, :, 1].mean()) > 200.0
    assert float(decoded[:, :, 0].mean()) < 40.0
    assert float(decoded[:, :, 2].mean()) < 40.0


def test_save_and_load_annotation_uses_clip_stem(monkeypatch, tmp_path) -> None:
    annotations_root = tmp_path / "clip_annotations"
    monkeypatch.setattr(clip_service, "_ANNOTATIONS_ROOT", annotations_root)

    empty = clip_service.get_annotation("clip_overlay_demo123_clip9_0.pt")
    assert empty["exists"] is False
    assert empty["content"] == ""
    assert empty["annotation_path"].endswith("clip_overlay_demo123_clip9_0.txt")

    saved = clip_service.save_annotation(
        "clip_overlay_demo123_clip9_0.pt",
        "frame 0 starts on a move; inspect source video pre-roll",
    )

    assert saved["exists"] is True
    assert saved["content"] == "frame 0 starts on a move; inspect source video pre-roll"
    saved_path = annotations_root / "clip_overlay_demo123_clip9_0.txt"
    assert saved_path.read_text() == saved["content"]

    loaded = clip_service.get_annotation("clip_overlay_demo123_clip9_0.pt")
    assert loaded == saved


def test_get_source_video_path_prepares_review_video(monkeypatch, tmp_path) -> None:
    source_video = tmp_path / "demo123.mp4"
    source_video.write_bytes(b"source-video")

    clip = {
        "frames": torch.zeros((1, 3, 8, 8), dtype=torch.uint8),
        "source_video_id": "demo123",
    }
    buffer = io.BytesIO()
    torch.save(clip, buffer)

    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool):
        calls.append(command)
        output_path = command[-1]
        assert output_path.endswith("source_review.mp4")
        with open(output_path, "wb") as handle:
            handle.write(b"review-video")

    monkeypatch.setattr(clip_service, "_get_video_path", lambda video_id: str(source_video))
    monkeypatch.setattr(clip_service.subprocess, "run", fake_run)

    session_id = clip_service.create_session(buffer.getvalue(), "clip_overlay_demo123_clip9_0.pt")
    try:
        review_path = clip_service.get_source_video_path(session_id)
    finally:
        clip_service.delete_session(session_id)

    assert review_path.endswith("source_review.mp4")
    assert calls and "+faststart" in calls[0]
