from __future__ import annotations

import chess
import torch
from api.services.annotate import physical_eval_service
from pipeline.shared import SQUARE_CLASS_NAMES

LABEL_INDEX_BY_NAME = {name: index for index, name in enumerate(SQUARE_CLASS_NAMES)}


def _square_index(square_name: str) -> int:
    file_index = ord(square_name[0]) - ord("a")
    rank_index = 8 - int(square_name[1])
    return rank_index * 8 + file_index


def test_list_clip_files_returns_relative_project_paths(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    clips_dir = project_root / "data" / "argus" / "train_real"
    clips_dir.mkdir(parents=True)
    (clips_dir / "clip_overlay_demo_clip12_0.pt").write_bytes(b"demo")

    monkeypatch.setattr(physical_eval_service, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        physical_eval_service.splits,
        "ensure_annotation_layout_migrated",
        lambda: None,
    )
    monkeypatch.setattr(
        physical_eval_service.splits,
        "get_source_video_split",
        lambda _video_id: None,
    )

    result = physical_eval_service.list_clip_files("data/argus/train_real")

    assert result["clips_dir"] == "data/argus/train_real"
    assert result["clips"][0]["clip_path"] == "data/argus/train_real/clip_overlay_demo_clip12_0.pt"
    assert result["clips"][0]["source_video_id"] == "demo"
    assert result["clips"][0]["clip_id"] == 12
    assert result["clips"][0]["annotated_frame_count"] == 0
    assert result["clips"][0]["num_frames"] is None
    assert result["clips"][0]["fully_annotated"] is False
    assert result["clips"][0]["assigned_split"] is None


def test_list_clip_files_marks_fully_annotated_clips(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    clips_dir = project_root / "data" / "argus" / "train_real"
    clips_dir.mkdir(parents=True)
    clip_path = clips_dir / "clip_overlay_demo_clip12_0.pt"
    clip_path.write_bytes(b"demo")

    monkeypatch.setattr(physical_eval_service, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        physical_eval_service.splits,
        "ensure_annotation_layout_migrated",
        lambda: None,
    )
    monkeypatch.setattr(
        physical_eval_service.splits,
        "get_source_video_split",
        lambda _video_id: "val",
    )
    monkeypatch.setattr(
        physical_eval_service.eval_dataset,
        "get_saved_frame_counts_by_clip",
        lambda: {"data/argus/train_real/clip_overlay_demo_clip12_0.pt": 5},
    )
    monkeypatch.setattr(physical_eval_service, "_get_clip_num_frames", lambda _path: 5)

    result = physical_eval_service.list_clip_files("data/argus/train_real")

    assert result["clips"][0]["annotated_frame_count"] == 5
    assert result["clips"][0]["num_frames"] == 5
    assert result["clips"][0]["fully_annotated"] is True
    assert result["clips"][0]["assigned_split"] == "val"


def test_get_move_corrections_infers_manual_move_from_saved_board(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data" / "argus" / "train_real").mkdir(parents=True)
    monkeypatch.setattr(physical_eval_service, "_PROJECT_ROOT", project_root)

    board = chess.Board()
    start_fen = board.fen()
    board.push(chess.Move.from_uci("e2e4"))
    after_e4 = board.fen()
    board.push(chess.Move.from_uci("e7e5"))

    clip_info = {
        "num_frames": 3,
        "frame_replay_fens": [start_fen, start_fen, after_e4],
        "frame_timestamps_seconds": [0.0, 0.5, 1.0],
        "moves": [
            {
                "frame_index": 1,
                "uci": "e2e4",
                "san": "e4",
                "detect_value": 1.0,
                "timestamp_seconds": 0.5,
                "estimated_otb_frame_index": None,
                "estimated_otb_timestamp_seconds": None,
                "side_to_move": "white",
                "fen_before": start_fen,
                "fen_after": after_e4,
            }
        ],
        "metadata": {"initial_board_fen": chess.Board().board_fen()},
    }
    annotations = [{"frame_index": 2, "labels": [None] * 64}]
    annotations[0]["labels"][_square_index("e5")] = LABEL_INDEX_BY_NAME["p"]

    monkeypatch.setattr(
        physical_eval_service.clip_service,
        "inspect",
        lambda _session_id: clip_info,
    )
    monkeypatch.setattr(
        physical_eval_service.eval_dataset,
        "list_board_annotations",
        lambda _clip_path: annotations,
    )

    result = physical_eval_service.get_move_corrections(
        "session-1",
        "data/argus/train_real/clip_overlay_demo_clip12_0.pt",
    )

    assert result["total_moves"] == 2
    assert [move["uci"] for move in result["moves"]] == ["e2e4", "e7e5"]
    assert result["moves"][1]["frame_index"] == 2
    assert result["moves"][1]["san"] == "e5"
    assert result["moves"][1]["is_manual"] is True
    assert result["frame_replay_fens"][2] == after_e4


def test_rectify_frame_reads_rgb_clip_tensor(monkeypatch) -> None:
    frames = torch.zeros((1, 3, 16, 16), dtype=torch.uint8)
    frames[0, 0] = 255

    monkeypatch.setattr(
        physical_eval_service.clip_service,
        "get_session",
        lambda _session_id: {"clip": {"frames": frames}},
    )

    result = physical_eval_service.rectify_frame(
        "session-1",
        0,
        corners=[[0, 0], [15, 0], [15, 15], [0, 15]],
        output_size=32,
    )

    assert result["output_size"] == 32
    assert isinstance(result["image_b64"], str)
    assert len(result["image_b64"]) > 20
