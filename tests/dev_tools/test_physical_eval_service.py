from __future__ import annotations

import torch
from api.services.annotate import physical_eval_service


def test_list_clip_files_returns_relative_project_paths(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "repo"
    clips_dir = project_root / "data" / "argus" / "train_real"
    clips_dir.mkdir(parents=True)
    (clips_dir / "clip_overlay_demo_clip12_0.pt").write_bytes(b"demo")

    monkeypatch.setattr(physical_eval_service, "_PROJECT_ROOT", project_root)

    result = physical_eval_service.list_clip_files("data/argus/train_real")

    assert result["clips_dir"] == "data/argus/train_real"
    assert result["clips"][0]["clip_path"] == "data/argus/train_real/clip_overlay_demo_clip12_0.pt"
    assert result["clips"][0]["source_video_id"] == "demo"
    assert result["clips"][0]["clip_id"] == 12


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
