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
        "ensure_source_video_splits_assigned",
        lambda _source_video_ids: {"demo": "val"},
    )

    result = physical_eval_service.list_clip_files("data/argus/train_real")

    assert result["clips_dir"] == "data/argus/train_real"
    assert result["clips"][0]["clip_path"] == "data/argus/train_real/clip_overlay_demo_clip12_0.pt"
    assert result["clips"][0]["source_video_id"] == "demo"
    assert result["clips"][0]["clip_id"] == 12
    assert result["clips"][0]["annotated_frame_count"] == 0
    assert result["clips"][0]["num_frames"] is None
    assert result["clips"][0]["fully_annotated"] is False
    assert result["clips"][0]["assigned_split"] == "val"


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
        "ensure_source_video_splits_assigned",
        lambda _source_video_ids: {"demo": "val"},
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

    # _get_clip_frame_rgb delegates to clip_service.get_camera_frame_rgb which
    # falls back to _get_stored_frame_rgb when source video is unavailable.
    # Provide a session via _sessions so the fallback path works.
    monkeypatch.setattr(
        physical_eval_service.clip_service,
        "_sessions",
        {"session-1": {"clip": {"frames": frames}, "filename": "test.pt"}},
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


def test_annotation_projection_context_returns_none_for_unusable_camera_bbox(monkeypatch) -> None:
    frames = torch.zeros((1, 3, 16, 16), dtype=torch.uint8)
    monkeypatch.setattr(
        physical_eval_service.clip_service,
        "_sessions",
        {"session-1": {"clip": {"frames": frames}, "filename": "test.pt"}},
    )
    monkeypatch.setattr(
        physical_eval_service.clip_service,
        "_get_overlay_preview_context",
        lambda _session: {
            "camera_bbox": (0, 0, 8, 8),
            "ref_resolution": (64, 64),
            "frame_indices": torch.tensor([0], dtype=torch.long),
            "video_path": "unused.mp4",
        },
    )

    result = physical_eval_service._annotation_projection_context_from_session(
        "session-1",
        0,
        padding_px=0,
    )

    assert result is None


def test_get_frame_annotation_projects_clip_space_corners_to_current_camera_crop(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data" / "argus" / "train_real").mkdir(parents=True)
    monkeypatch.setattr(physical_eval_service, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        physical_eval_service,
        "_annotation_projection_context_from_session",
        lambda _session_id, _frame_index, *, padding_px: (
            physical_eval_service._AnnotationProjectionContext(
                clip_frame_size=(224, 224),
                zero_padding_bbox=(100, 200, 448, 224),
                current_bbox=(90, 190, 468, 244) if padding_px == 10 else (100, 200, 448, 224),
                source_frame_index=7,
            )
        ),
    )

    annotation = physical_eval_service._get_frame_annotation(
        type(
            "FakeDataset",
            (),
            {
                "load_board_annotation": staticmethod(
                    lambda _clip_path, _frame_index: {
                        "annotation_id": "demo_frame0003",
                        "clip_path": "data/argus/train_real/clip_overlay_demo_clip12_0.pt",
                        "frame_index": 3,
                        "corners": [[0.0, 0.0], [224.0, 0.0], [224.0, 224.0], [0.0, 224.0]],
                        "labels": [None] * 64,
                        "labeled_square_count": 0,
                        "rectified_board_path": "data/physical/val/boards/demo_frame0003.jpg",
                        "rectified_size": 512,
                        "created_at": "2026-04-15T00:00:00+00:00",
                    }
                ),
            },
        ),
        "data/argus/train_real/clip_overlay_demo_clip12_0.pt",
        3,
        session_id="session-1",
        padding_px=10,
    )

    assert annotation is not None
    assert annotation["corners"] == [[10.0, 10.0], [458.0, 10.0], [458.0, 234.0], [10.0, 234.0]]
    assert annotation["clip_frame_size"] == [224, 224]
    assert annotation["source_frame_index"] == 7


def test_save_annotation_stores_clip_space_corners_but_returns_display_corners(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data" / "argus" / "train_real").mkdir(parents=True)
    clip_path = project_root / "data" / "argus" / "train_real" / "clip_overlay_demo_clip12_0.pt"
    clip_path.write_bytes(b"demo")
    monkeypatch.setattr(physical_eval_service, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        physical_eval_service,
        "_get_clip_frame_rgb",
        lambda _session_id, _frame_index, padding_px=0: torch.zeros(
            (244, 468, 3), dtype=torch.uint8
        ).numpy(),
    )
    monkeypatch.setattr(
        physical_eval_service,
        "_annotation_projection_context_from_session",
        lambda _session_id, _frame_index, *, padding_px: (
            physical_eval_service._AnnotationProjectionContext(
                clip_frame_size=(224, 224),
                zero_padding_bbox=(100, 200, 448, 224),
                current_bbox=(90, 190, 468, 244) if padding_px == 10 else (100, 200, 448, 224),
                source_frame_index=7,
            )
        ),
    )

    captured: dict[str, object] = {}

    class FakeDataset:
        @staticmethod
        def save_board_annotation(image_rgb, **kwargs):
            captured["image_shape"] = image_rgb.shape
            captured.update(kwargs)
            return {
                "annotation_id": "demo_frame0003",
                "clip_path": kwargs["clip_path"],
                "frame_index": kwargs["frame_index"],
                "source_video_id": kwargs["source_video_id"],
                "corners": kwargs["corners"],
                "labels": kwargs["labels"],
                "labeled_square_count": 0,
                "rectified_board_path": "data/physical/val/boards/demo_frame0003.jpg",
                "rectified_size": kwargs["output_size"],
                "created_at": "2026-04-15T00:00:00+00:00",
                "corner_space": kwargs["corner_space"],
                "clip_frame_size": kwargs["clip_frame_size"],
                "native_corners": kwargs["native_corners"],
                "native_image_bbox": kwargs["native_image_bbox"],
                "source_frame_index": kwargs["source_frame_index"],
            }

        @staticmethod
        def get_annotation_summary():
            return {"board_annotation_count": 1}

    result = physical_eval_service._save_annotation(
        FakeDataset,
        "session-1",
        "data/argus/train_real/clip_overlay_demo_clip12_0.pt",
        3,
        [[10.0, 10.0], [458.0, 10.0], [458.0, 234.0], [10.0, 234.0]],
        [None] * 64,
        output_size=512,
        padding_px=10,
    )

    assert captured["clip_path"] == "data/argus/train_real/clip_overlay_demo_clip12_0.pt"
    assert captured["corners"] == [[0.0, 0.0], [224.0, 0.0], [224.0, 224.0], [0.0, 224.0]]
    assert captured["image_corners"] == [[10.0, 10.0], [458.0, 10.0], [458.0, 234.0], [10.0, 234.0]]
    assert captured["clip_frame_size"] == [224, 224]
    assert captured["native_corners"] == [
        [10.0, 10.0],
        [458.0, 10.0],
        [458.0, 234.0],
        [10.0, 234.0],
    ]
    assert captured["native_image_bbox"] == [90, 190, 468, 244]
    assert captured["source_frame_index"] == 7
    assert captured["corner_space"] == "clip_frame"
    assert captured["image_shape"] == (244, 468, 3)

    assert result["annotation"]["corners"] == [
        [10.0, 10.0],
        [458.0, 10.0],
        [458.0, 234.0],
        [10.0, 234.0],
    ]
    assert result["summary"] == {"board_annotation_count": 1}
