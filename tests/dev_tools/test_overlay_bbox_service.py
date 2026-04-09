"""Tests for overlay bbox annotation frame selection and path resolution."""

import json
from pathlib import Path

import cv2
import numpy as np
from api.services.annotate import overlay_bbox_service
from pipeline.overlay.scanner import OverlayDetection


def _write_image(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _configure_paths(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    videos_dir = tmp_path / "videos"
    ground_truth_path = tmp_path / "ground_truth.json"
    targets_path = tmp_path / "annotation_targets.json"

    monkeypatch.setattr(overlay_bbox_service, "VIDEOS_DIR", videos_dir)
    monkeypatch.setattr(overlay_bbox_service, "GROUND_TRUTH_PATH", ground_truth_path)
    monkeypatch.setattr(
        overlay_bbox_service,
        "_LEGACY_FRAMES_DIR",
        tmp_path / "legacy_frames",
    )
    monkeypatch.setattr(
        overlay_bbox_service,
        "_LEGACY_GT_PATH",
        tmp_path / "legacy_ground_truth.json",
    )
    monkeypatch.setattr(
        overlay_bbox_service,
        "_FIXTURE_TARGETS_PATH",
        targets_path,
    )
    monkeypatch.setattr(
        overlay_bbox_service,
        "_frame_path",
        lambda video_id, tier, label: videos_dir / video_id / tier / f"{label}.jpg",
    )
    overlay_bbox_service._invalidate_frame_cache()
    return videos_dir, ground_truth_path, targets_path


def test_list_frames_prioritizes_targets_and_includes_fullres_only(tmp_path, monkeypatch) -> None:
    videos_dir, ground_truth_path, targets_path = _configure_paths(tmp_path, monkeypatch)

    _write_image(videos_dir / "alpha" / "fullres" / "25pct.jpg", 1920, 1080)
    _write_image(videos_dir / "beta" / "hires" / "25pct.jpg", 1280, 720)
    _write_image(videos_dir / "gamma" / "hires" / "25pct.jpg", 1280, 720)

    ground_truth_path.write_text(
        json.dumps(
            {
                "beta/25pct": {
                    "has_overlay": True,
                    "bbox": [10, 20, 30, 30],
                    "frame_width": 1280,
                    "frame_height": 720,
                    "annotated_at": "2026-04-07T00:00:00+00:00",
                }
            }
        )
    )
    targets_path.write_text(
        json.dumps(
            [
                {"key": "alpha/25pct", "issue": "bbox slightly off"},
                {"key": "beta/25pct", "issue": "false negative"},
            ]
        )
    )

    result = overlay_bbox_service.list_frames()
    frames = result["frames"]

    assert [frame["key"] for frame in frames[:2]] == ["alpha/25pct", "beta/25pct"]
    assert frames[0]["is_target"] is True
    assert frames[0]["target_issue"] == "bbox slightly off"
    assert frames[0]["annotated"] is False
    assert frames[1]["annotated"] is True
    assert any(frame["key"] == "alpha/25pct" for frame in frames)


def test_get_frame_path_preserves_annotated_tier_and_falls_back_to_fullres(
    tmp_path,
    monkeypatch,
) -> None:
    videos_dir, ground_truth_path, _targets_path = _configure_paths(tmp_path, monkeypatch)

    hires_path = videos_dir / "video1" / "hires" / "25pct.jpg"
    fullres_path = videos_dir / "video1" / "fullres" / "25pct.jpg"
    _write_image(hires_path, 1280, 720)
    _write_image(fullres_path, 1920, 1080)
    fullres_only_path = videos_dir / "video2" / "fullres" / "50pct.jpg"
    _write_image(fullres_only_path, 1920, 1080)

    ground_truth_path.write_text(
        json.dumps(
            {
                "video1/25pct": {
                    "has_overlay": True,
                    "bbox": [1, 2, 3, 4],
                    "frame_width": 1280,
                    "frame_height": 720,
                    "annotated_at": "2026-04-07T00:00:00+00:00",
                }
            }
        )
    )

    assert overlay_bbox_service.get_frame_path("video1", "25pct") == hires_path
    assert overlay_bbox_service.get_frame_path("video2", "50pct") == fullres_only_path


def test_auto_detect_bbox_returns_refined_detector_candidate(tmp_path, monkeypatch) -> None:
    videos_dir, _ground_truth_path, _targets_path = _configure_paths(tmp_path, monkeypatch)

    frame_path = videos_dir / "video1" / "hires" / "25pct.jpg"
    _write_image(frame_path, 1280, 720)

    monkeypatch.setattr(
        overlay_bbox_service,
        "runtime_overlay_check",
        lambda _frame: OverlayDetection(
            found=True,
            bbox=(100, 120, 320, 320),
            seed_bbox=(100, 120, 320, 320),
            score=0.91,
            frame_resolution=(1280, 720),
        ),
    )
    monkeypatch.setattr(
        overlay_bbox_service,
        "_refine_alignment",
        lambda _frame, bbox, max_shift=5: bbox,
    )
    monkeypatch.setattr(overlay_bbox_service, "compute_grid_regularity", lambda _region: 0.75)
    monkeypatch.setattr(overlay_bbox_service, "check_alternating_pattern", lambda _region: True)

    result = overlay_bbox_service.auto_detect_bbox(frame_path)

    assert result == {
        "detected": True,
        "bbox": [100, 120, 320, 320],
        "score": 0.91,
        "grid_score": 0.75,
        "has_pattern": True,
        "detector_bbox": [100, 120, 320, 320],
    }


def test_auto_detect_bbox_reports_missing_overlay(tmp_path, monkeypatch) -> None:
    videos_dir, _ground_truth_path, _targets_path = _configure_paths(tmp_path, monkeypatch)

    frame_path = videos_dir / "video1" / "hires" / "25pct.jpg"
    _write_image(frame_path, 1280, 720)

    monkeypatch.setattr(
        overlay_bbox_service,
        "runtime_overlay_check",
        lambda _frame: OverlayDetection(found=False, frame_resolution=(1280, 720)),
    )

    assert overlay_bbox_service.auto_detect_bbox(frame_path) == {
        "detected": False,
        "bbox": None,
        "score": 0.0,
    }
