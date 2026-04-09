"""Tests for real-footage data inventory and processing service."""

from __future__ import annotations

import sys
import time
import types

from api.services.data import real_service


def setup_function() -> None:
    real_service._current_job = None
    real_service._cancel_event = real_service.threading.Event()


def test_get_overview_classifies_ready_blocked_and_processed(monkeypatch) -> None:
    monkeypatch.setattr(
        real_service,
        "_list_local_videos",
        lambda max_file_size_mb: [
            {
                "video_id": "READYVIDEO1",
                "video_path": "/tmp/READYVIDEO1.mp4",
                "file_size_mb": 120.0,
                "modified_ts": 3.0,
            },
            {
                "video_id": "BLOCKVIDEO1",
                "video_path": "/tmp/BLOCKVIDEO1.mp4",
                "file_size_mb": 130.0,
                "modified_ts": 2.0,
            },
            {
                "video_id": "DONEVIDEO11",
                "video_path": "/tmp/DONEVIDEO11.mp4",
                "file_size_mb": 140.0,
                "modified_ts": 1.0,
            },
        ],
    )
    monkeypatch.setattr(
        real_service,
        "_load_video_rows",
        lambda video_ids: {
            "READYVIDEO1": {
                "video_id": "READYVIDEO1",
                "channel_handle": "@ready",
                "title": "Ready video",
                "published_at": None,
                "screening_status": "approved",
                "layout_type": "overlay",
            },
            "BLOCKVIDEO1": {
                "video_id": "BLOCKVIDEO1",
                "channel_handle": "@blocked",
                "title": "Blocked video",
                "published_at": None,
                "screening_status": "rejected",
                "layout_type": "overlay",
            },
            "DONEVIDEO11": {
                "video_id": "DONEVIDEO11",
                "channel_handle": "@done",
                "title": "Done video",
                "published_at": None,
                "screening_status": "approved",
                "layout_type": "overlay",
            },
        },
    )
    monkeypatch.setattr(real_service, "_load_db_clip_counts", lambda video_ids: {"DONEVIDEO11": 1})
    monkeypatch.setattr(real_service, "_load_calibrated_channels", lambda: {"@ready", "@done"})
    monkeypatch.setattr(
        real_service,
        "_load_existing_clip_counts",
        lambda clips_dir: {"DONEVIDEO11": 2},
    )
    monkeypatch.setattr(
        real_service.synthetic_service,
        "get_clip_stats",
        lambda clips_dir: {
            "clip_count": 2,
            "total_frames": 300,
            "avg_frames_per_clip": 150.0,
            "total_moves": 20,
            "avg_moves_per_clip": 10.0,
            "moves_per_clip_distribution": [10, 10],
            "avg_file_size_mb": 1.5,
            "total_size_mb": 3.0,
            "avg_legal_moves": 35.0,
            "image_size": [224, 224],
            "clip_length": 200,
        },
    )

    overview = real_service.get_overview(limit=10)

    assert overview["ready_video_count"] == 1
    assert overview["processed_video_count"] == 1
    assert overview["blocked_video_count"] == 1
    assert overview["source_video_count"] == 1

    videos = {video["video_id"]: video for video in overview["videos"]}
    assert videos["READYVIDEO1"]["ready"] is True
    assert videos["READYVIDEO1"]["blocker"] is None
    assert videos["BLOCKVIDEO1"]["blocker"] == "not_approved"
    assert videos["DONEVIDEO11"]["blocker"] == "already_processed"


def test_start_processing_runs_generate_from_video_in_background(monkeypatch) -> None:
    monkeypatch.setattr(
        real_service,
        "get_overview",
        lambda clips_dir, max_file_size_mb=200.0, limit=5000: {
            "videos": [
                {
                    "video_id": "READYVIDEO1",
                    "video_path": "/tmp/READYVIDEO1.mp4",
                    "channel_handle": "@ready",
                    "title": "Ready video",
                    "ready": True,
                },
                {
                    "video_id": "BLOCKVIDEO1",
                    "video_path": "/tmp/BLOCKVIDEO1.mp4",
                    "channel_handle": "@blocked",
                    "title": "Blocked video",
                    "ready": False,
                },
            ]
        },
    )

    calls: list[tuple[str, str, str, int]] = []

    def fake_generate_from_video(video_path, channel_handle, output_dir, min_moves_per_segment):
        calls.append((video_path, channel_handle, output_dir, min_moves_per_segment))
        return [
            {"filepath": f"{output_dir}/clip_a.pt", "num_frames": 120, "num_moves": 8},
            {"filepath": f"{output_dir}/clip_b.pt", "num_frames": 140, "num_moves": 9},
        ]

    monkeypatch.setitem(
        sys.modules,
        "pipeline.overlay.overlay_clip_generator",
        types.SimpleNamespace(generate_from_video=fake_generate_from_video),
    )

    status = real_service.start_processing(limit=10, clips_dir="data/argus/train_real")
    assert status["status"] in {"running", "done"}
    assert status["total_videos"] == 1

    for _ in range(100):
        status = real_service.get_processing_status()
        if status["status"] == "done":
            break
        time.sleep(0.01)

    assert status["status"] == "done"
    assert status["generated_clips"] == 2
    assert status["completed_videos"] == 1
    assert status["results"] == [
        {
            "video_id": "READYVIDEO1",
            "title": "Ready video",
            "status": "generated",
            "generated_clip_count": 2,
            "error": None,
        }
    ]
    assert calls == [
        (
            "/tmp/READYVIDEO1.mp4",
            "@ready",
            str(real_service._resolve("data/argus/train_real")),
            5,
        )
    ]


def test_start_processing_rejects_when_no_videos_are_ready(monkeypatch) -> None:
    monkeypatch.setattr(
        real_service,
        "get_overview",
        lambda clips_dir, max_file_size_mb=200.0, limit=5000: {"videos": []},
    )

    try:
        real_service.start_processing(limit=10)
    except ValueError as e:
        assert str(e) == "No eligible local videos found to process."
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError")
