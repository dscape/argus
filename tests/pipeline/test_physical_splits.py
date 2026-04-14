from __future__ import annotations

import json

from pipeline.physical import splits


def test_ensure_annotation_layout_migrated_moves_legacy_eval_into_val(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "repo"
    data_root = project_root / "data" / "physical"
    legacy_eval_root = data_root / "eval"
    legacy_eval_root.mkdir(parents=True)
    (legacy_eval_root / "boards").mkdir()
    (legacy_eval_root / "squares").mkdir()
    (legacy_eval_root / "boards" / "sample.jpg").write_bytes(b"board")
    (legacy_eval_root / "squares" / "sample_a8.jpg").write_bytes(b"square")

    (legacy_eval_root / "board_annotations.jsonl").write_text(
        json.dumps(
            {
                "annotation_id": "ann-1",
                "clip_path": "data/argus/train_real/clip_overlay_demo_clip1_0.pt",
                "frame_index": 3,
                "source_video_id": "demo",
                "rectified_board_path": "data/physical/eval/boards/sample.jpg",
            },
            sort_keys=True,
        )
        + "\n"
    )
    (legacy_eval_root / "square_manifest.jsonl").write_text(
        json.dumps(
            {
                "annotation_id": "ann-1",
                "square_index": 0,
                "source_video_id": "demo",
                "crop_path": "data/physical/eval/squares/sample_a8.jpg",
                "split": "eval_holdout",
            },
            sort_keys=True,
        )
        + "\n"
    )

    monkeypatch.setattr(splits, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(splits, "_DATA_ROOT", data_root)
    monkeypatch.setattr(splits, "_SOURCE_VIDEO_SPLITS_PATH", data_root / "source_video_splits.json")
    monkeypatch.setattr(
        splits,
        "_CANONICAL_ROOTS",
        {"train": data_root / "train", "val": data_root / "val"},
    )
    monkeypatch.setattr(
        splits,
        "_LEGACY_ROOTS",
        {"train": data_root / "train_manual", "val": data_root / "eval"},
    )

    splits.ensure_annotation_layout_migrated()

    canonical_val_root = data_root / "val"
    assert canonical_val_root.exists()
    assert not legacy_eval_root.exists()

    board_rows = [
        json.loads(line)
        for line in (canonical_val_root / "board_annotations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert board_rows[0]["rectified_board_path"] == "data/physical/val/boards/sample.jpg"

    square_rows = [
        json.loads(line)
        for line in (canonical_val_root / "square_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert square_rows[0]["crop_path"] == "data/physical/val/squares/sample_a8.jpg"
    assert square_rows[0]["split"] == "val"

    split_manifest = json.loads((data_root / "source_video_splits.json").read_text())
    assert split_manifest["source_video_splits"] == {"demo": "val"}
