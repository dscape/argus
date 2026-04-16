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

    image_path = study_dir / "frames" / "001_clip_a_f0002.png"
    _write_image(image_path)

    manifest = [
        {
            "selected_index": 1,
            "annotation_id": "clip_a_frame0002",
            "clip_path": "data/argus/train_real/clip_a.pt",
            "clip_filename": "clip_a.pt",
            "frame_index": 2,
            "source_video_id": "video-a",
            "suggested_root_cause": "rectification_or_classifier",
            "decoded_error_count": 2,
            "stateless_error_count": 7,
            "decoded_matches_previous_gt": False,
            "decoded_matches_next_gt": False,
            "decoded_move_uci": "e2e4",
            "decoded_error_squares": ["e4", "e5"],
            "decoded_changed_squares": ["e2", "e4"],
            "gt_changed_squares": ["e2", "e4"],
            "legal_from_previous_decoded": {
                "best_legal_matches_gt": True,
                "best_legal_move_uci": "e2e4",
                "gt_legal_rank": 1,
            },
            "image_path": "outputs/bundle_a/frames/001_clip_a_f0002.png",
        }
    ]
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with (study_dir / "manual_buckets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selected_index",
                "annotation_id",
                "clip_path",
                "frame_index",
                "suggested_root_cause",
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "selected_index": 1,
                "annotation_id": "clip_a_frame0002",
                "clip_path": "data/argus/train_real/clip_a.pt",
                "frame_index": 2,
                "suggested_root_cause": "rectification_or_classifier",
                "decoded_error_count": 2,
                "stateless_error_count": 7,
                "decoded_matches_previous_gt": "False",
                "decoded_matches_next_gt": "False",
                "best_legal_matches_gt": "True",
                "best_legal_move_uci": "e2e4",
                "gt_legal_rank": 1,
                "image_path": "outputs/bundle_a/frames/001_clip_a_f0002.png",
                "final_bucket": "piece classifier / square evidence",
                "notes": "Knight vs bishop confusion.",
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
                "selected_failures": 1,
                "total_failures": 7,
                "sample_mode": "round_robin",
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

    detail = service.get_failure_study("outputs/bundle_a")
    assert detail["report_metrics"]["board_exact_match"] == 0.2165
    assert detail["entries"][0]["final_bucket"] == "piece classifier / square evidence"
    assert detail["entries"][0]["best_legal_matches_gt"] is True
    assert detail["bucket_counts"] == {"piece classifier / square evidence": 1}


def test_get_failure_study_context(monkeypatch, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(service, "_OUTPUTS_DIR", tmp_path / "outputs")

    rows = [
        SimpleNamespace(
            annotation_id=f"clip_a_frame000{i}",
            clip_path="data/argus/train_real/clip_a.pt",
            frame_index=i,
            board_path=f"boards/frame{i}.png",
        )
        for i in range(3)
    ]
    monkeypatch.setattr(
        service.failure_study,
        "_load_rows_by_clip",
        lambda observation_input: {"data/argus/train_real/clip_a.pt": rows},
    )
    monkeypatch.setattr(
        service.failure_study,
        "_load_row_image",
        lambda row, observation_input, clip_cache: np.full(
            (12, 12, 3),
            int(row.frame_index),
            dtype=np.uint8,
        ),
    )

    context = service.get_failure_study_context(
        "outputs/bundle_a",
        selected_index=1,
        context_frames=2,
        image_max_side=64,
    )
    assert context["selected_index"] == 1
    assert len(context["frames"]) == 3
    assert context["frames"][-1]["is_anchor"] is True
    assert context["frames"][-1]["frame_index"] == 2
    assert context["anchor_panel_data_url"].startswith("data:image/png;base64,")


def test_update_failure_study_entry(monkeypatch, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(service, "_OUTPUTS_DIR", tmp_path / "outputs")

    updated = service.update_failure_study_entry(
        "outputs/bundle_a",
        selected_index=1,
        final_bucket="decoder / wrong legal hypothesis / error propagation",
        notes="Stayed on the wrong legal successor after the first miss.",
    )

    assert updated["selected_index"] == 1
    assert updated["final_bucket"] == "decoder / wrong legal hypothesis / error propagation"

    csv_text = (tmp_path / "outputs" / "bundle_a" / "manual_buckets.csv").read_text()
    assert "Stayed on the wrong legal successor after the first miss." in csv_text
