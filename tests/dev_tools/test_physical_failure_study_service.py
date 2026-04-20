from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from api.services.evaluate import physical_failure_study_service as service
from PIL import Image


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(32, 64, 96)).save(path)


def _write_bundle(tmp_path: Path) -> Path:
    study_dir = tmp_path / "outputs" / "bundle_a"
    study_dir.mkdir(parents=True)

    failing_image_path = study_dir / "episodes" / "ep001" / "frame_0002.png"
    preceding_image_path = study_dir / "episodes" / "ep001" / "frame_0001.png"
    _write_image(failing_image_path)
    _write_image(preceding_image_path)

    manifest = [
        {
            "episode_id": "ep001",
            "selected_index": 1,
            "source_video_id": "video-a",
            "clip_path": "data/argus/train_real/clip_a.pt",
            "clip_filename": "clip_a.pt",
            "first_frame_index": 2,
            "last_frame_index": 4,
            "length": 3,
            "preceding_frame_count": 1,
            "suggested_bucket": "piece classifier / square evidence",
            "failing_frame": {
                "annotation_id": "clip_a_frame0002",
                "clip_path": "data/argus/train_real/clip_a.pt",
                "clip_filename": "clip_a.pt",
                "frame_index": 2,
                "source_video_id": "video-a",
                "gt_fen": "8/8/8/8/8/8/8/8",
                "decoded_fen": "8/8/8/8/8/8/8/8",
                "decoded_full_fen": "8/8/8/8/8/8/8/8 w - - 0 1",
                "stateless_fen": "8/8/8/8/8/8/8/8",
                "decoded_move_uci": "e2e4",
                "decoded_error_count": 2,
                "stateless_error_count": 7,
                "decoded_matches_previous_gt": False,
                "decoded_matches_next_gt": False,
                "decoded_error_squares": ["e4", "e5"],
                "decoded_changed_squares": ["e2", "e4"],
                "gt_changed_squares": ["e2", "e4"],
                "offset_from_failure": 0,
                "is_failing_frame": True,
                "legal_from_previous_decoded": {
                    "best_legal_matches_gt": True,
                    "best_legal_move_uci": "e2e4",
                    "gt_legal_rank": 1,
                },
                "image_path": "outputs/bundle_a/episodes/ep001/frame_0002.png",
            },
            "preceding_frames": [
                {
                    "annotation_id": "clip_a_frame0001",
                    "clip_path": "data/argus/train_real/clip_a.pt",
                    "clip_filename": "clip_a.pt",
                    "frame_index": 1,
                    "source_video_id": "video-a",
                    "gt_fen": "8/8/8/8/8/8/8/8",
                    "decoded_fen": "8/8/8/8/8/8/8/8",
                    "decoded_full_fen": "8/8/8/8/8/8/8/8 w - - 0 1",
                    "stateless_fen": "8/8/8/8/8/8/8/8",
                    "decoded_error_count": 0,
                    "stateless_error_count": 0,
                    "decoded_matches_previous_gt": False,
                    "decoded_matches_next_gt": False,
                    "offset_from_failure": -1,
                    "is_failing_frame": False,
                    "image_path": "outputs/bundle_a/episodes/ep001/frame_0001.png",
                }
            ],
        }
    ]
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with (study_dir / "manual_buckets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selected_index",
                "episode_id",
                "source_video_id",
                "clip_path",
                "first_frame_index",
                "last_frame_index",
                "length",
                "suggested_bucket",
                "decoded_error_count",
                "stateless_error_count",
                "decoded_matches_previous_gt",
                "decoded_matches_next_gt",
                "best_legal_matches_gt",
                "best_legal_move_uci",
                "gt_legal_rank",
                "image_path",
                "final_bucket",
                "notes",
                "bucket_updated_at",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "selected_index": 1,
                "episode_id": "ep001",
                "source_video_id": "video-a",
                "clip_path": "data/argus/train_real/clip_a.pt",
                "first_frame_index": 2,
                "last_frame_index": 4,
                "length": 3,
                "suggested_bucket": "piece classifier / square evidence",
                "decoded_error_count": 2,
                "stateless_error_count": 7,
                "decoded_matches_previous_gt": "False",
                "decoded_matches_next_gt": "False",
                "best_legal_matches_gt": "True",
                "best_legal_move_uci": "e2e4",
                "gt_legal_rank": 1,
                "image_path": "outputs/bundle_a/episodes/ep001/frame_0002.png",
                "final_bucket": "piece classifier / square evidence",
                "notes": "Knight vs bishop confusion.",
                "bucket_updated_at": "2026-04-16T00:00:00+00:00",
            }
        )

    report_path = tmp_path / "outputs" / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "board_exact_match": 0.2165,
                    "non_empty_accuracy": 0.864,
                    "macro_f1": 0.8166,
                    "accuracy": 0.9026,
                },
                "move_detection_recall": 0.2969,
                "static_frame_false_change_rate": 0.0181,
            }
        )
    )

    (study_dir / "summary.json").write_text(
        json.dumps(
            {
                "manifest": "outputs/bundle_a/manifest.json",
                "manual_buckets_csv": "outputs/bundle_a/manual_buckets.csv",
                "report_path": "outputs/report.json",
                "selected_episodes": 1,
                "total_episodes": 7,
                "config": {
                    "observation_input": "rectified_board",
                    "tracker_mode": "lookahead",
                    "lookahead_window": 3,
                    "lookahead_margin": 8.0,
                },
            },
            indent=2,
        )
    )
    return study_dir


def test_list_and_load_failure_study(monkeypatch, tmp_path: Path) -> None:
    study_dir = _write_bundle(tmp_path)
    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(service, "_OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(service, "_candidate_summary_paths", lambda: [study_dir / "summary.json"])

    studies = service.list_failure_studies()
    assert len(studies) == 1
    assert studies[0]["path"] == "outputs/bundle_a"
    assert studies[0]["selected_episodes"] == 1
    assert studies[0]["total_episodes"] == 7

    detail = service.get_failure_study("outputs/bundle_a")
    assert detail["report_metrics"]["board_exact_match"] == 0.2165
    assert detail["entries"][0]["final_bucket"] == "piece classifier / square evidence"
    assert (
        detail["entries"][0]["failing_frame"]["legal_from_previous_decoded"][
            "best_legal_matches_gt"
        ]
        is True
    )
    assert detail["entries"][0]["episode_id"] == "ep001"
    assert detail["bucket_counts"] == {"piece classifier / square evidence": 1}


def test_resolve_image_path_and_export_csv(monkeypatch, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(service, "_OUTPUTS_DIR", tmp_path / "outputs")

    resolved = service.resolve_image_path(
        "outputs/bundle_a",
        "outputs/bundle_a/episodes/ep001/frame_0002.png",
    )
    assert resolved.name == "frame_0002.png"

    csv_payload = service.export_manual_buckets_csv("outputs/bundle_a")
    assert "episode_id" in csv_payload
    assert "ep001" in csv_payload


def test_update_failure_study_entry(monkeypatch, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(service, "_OUTPUTS_DIR", tmp_path / "outputs")

    updated = service.update_failure_study_entry(
        "outputs/bundle_a",
        episode_id="ep001",
        final_bucket="decoder / wrong legal hypothesis / error propagation",
        notes="Stayed on the wrong legal successor after the first miss.",
    )

    assert updated["episode_id"] == "ep001"
    assert updated["final_bucket"] == "decoder / wrong legal hypothesis / error propagation"
    assert updated["updated_at"]

    csv_text = (tmp_path / "outputs" / "bundle_a" / "manual_buckets.csv").read_text()
    assert "Stayed on the wrong legal successor after the first miss." in csv_text


def test_get_failure_study_adds_geometry_and_probabilities(
    monkeypatch,
    tmp_path: Path,
) -> None:
    study_dir = _write_bundle(tmp_path)
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    square_probabilities = [[0.0] * 13 for _ in range(64)]
    square_probabilities[0][1] = 0.9
    manifest[0]["failing_frame"]["stateless_square_probabilities"] = square_probabilities
    manifest_path.write_text(json.dumps(manifest, indent=2))

    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(service, "_OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(
        service,
        "_annotation_rows_by_id",
        lambda: {
            "clip_a_frame0002": SimpleNamespace(
                corners=((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)),
            )
        },
    )
    monkeypatch.setattr(
        service,
        "load_annotated_board_frame_bgr",
        lambda row, clip_cache: np.zeros((100, 100, 3), dtype=np.uint8),
    )

    detail = service.get_failure_study("outputs/bundle_a")
    frame = detail["entries"][0]["failing_frame"]

    assert frame["stateless_square_probabilities"][0][1] == 0.9
    assert frame["raw_image_path"].endswith("clip_a_frame0002.png")
    assert len(frame["geometry_square_quads"]["a8"]) == 4
    assert len(frame["geometry_piece_bboxes"]["a8"]) == 4
