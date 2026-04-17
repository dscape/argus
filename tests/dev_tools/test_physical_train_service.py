from __future__ import annotations

from api.services.annotate import physical_train_service


def test_list_clip_files_excludes_held_out_videos(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    clips_dir = project_root / "data" / "argus" / "train_real"
    clips_dir.mkdir(parents=True)
    (clips_dir / "clip_overlay_demo_clip12_0.pt").write_bytes(b"demo")
    (clips_dir / "clip_overlay_heldout_clip99_0.pt").write_bytes(b"heldout")

    monkeypatch.setattr(physical_train_service.physical_eval_service, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        physical_train_service.physical_eval_service.splits,
        "ensure_annotation_layout_migrated",
        lambda: None,
    )
    monkeypatch.setattr(
        physical_train_service.physical_eval_service.splits,
        "ensure_source_video_splits_assigned",
        lambda _source_video_ids: {"demo": "train", "heldout": "val"},
    )
    monkeypatch.setattr(
        physical_train_service.manual_train_dataset,
        "get_saved_frame_counts_by_clip",
        lambda: {},
    )
    monkeypatch.setattr(
        physical_train_service.manual_train_dataset,
        "load_transient_annotation",
        lambda _clip_path: None,
    )

    result = physical_train_service.list_clip_files("data/argus/train_real")

    assert [clip["source_video_id"] for clip in result["clips"]] == ["demo"]
    assert result["clips"][0]["clip_path"] == "data/argus/train_real/clip_overlay_demo_clip12_0.pt"
    assert result["clips"][0]["transient_annotation_complete"] is False
