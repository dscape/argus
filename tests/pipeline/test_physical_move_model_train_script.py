from __future__ import annotations

from types import SimpleNamespace

import pytest
import scripts.eval_physical_move_model as eval_physical_move_model
import scripts.train_physical_move_model as train_physical_move_model
from scripts.train_physical_move_model import (
    normalized_checkpoint_model_config,
    resolve_selection_sequence_source,
    select_real_val_source_video_ids,
)


def test_eval_move_model_normalized_checkpoint_model_config_remaps_legacy_siglip2_to_siglip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eval_physical_move_model,
        "load_transformers_config",
        lambda _model_name: SimpleNamespace(model_type="siglip"),
    )

    normalized = eval_physical_move_model.normalized_checkpoint_model_config(
        {
            "vision_encoder_type": "siglip2",
            "vision_encoder_name": "google/siglip2-base-patch16-224",
            "temporal_d_model": 256,
        }
    )

    assert normalized["vision_encoder_type"] == "siglip"
    assert normalized["vision_encoder_name"] == "google/siglip2-base-patch16-224"
    assert normalized["temporal_d_model"] == 256


def test_normalized_checkpoint_model_config_remaps_legacy_siglip2_to_siglip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_physical_move_model,
        "load_transformers_config",
        lambda _model_name: SimpleNamespace(model_type="siglip"),
    )

    normalized = normalized_checkpoint_model_config(
        {
            "vision_encoder_type": "siglip2",
            "vision_encoder_name": "google/siglip2-base-patch16-224",
            "temporal_d_model": 256,
        }
    )

    assert normalized["vision_encoder_type"] == "siglip"
    assert normalized["vision_encoder_name"] == "google/siglip2-base-patch16-224"
    assert normalized["temporal_d_model"] == 256


def test_resolve_selection_sequence_source_auto_prefers_internal_real_val() -> None:
    resolved = resolve_selection_sequence_source(
        "auto",
        use_real=True,
        real_val_source_video_ids={"video123"},
    )

    assert resolved == "real_val"


def test_resolve_selection_sequence_source_real_val_requires_real_data() -> None:
    with pytest.raises(ValueError, match="requires --use-real"):
        resolve_selection_sequence_source(
            "real_val",
            use_real=False,
            real_val_source_video_ids={"video123"},
        )


def test_select_real_val_source_video_ids_prefers_largest_clip_coverage() -> None:
    rows = [
        SimpleNamespace(source_video_id="videoA", clip_path="a1.pt"),
        SimpleNamespace(source_video_id="videoA", clip_path="a1.pt"),
        SimpleNamespace(source_video_id="videoA", clip_path="a2.pt"),
        SimpleNamespace(source_video_id="videoB", clip_path="b1.pt"),
        SimpleNamespace(source_video_id="videoB", clip_path="b2.pt"),
        SimpleNamespace(source_video_id="videoB", clip_path="b3.pt"),
        SimpleNamespace(source_video_id="videoC", clip_path="c1.pt"),
    ]

    selected = select_real_val_source_video_ids(
        rows,
        requested_source_video_ids=[],
        requested_count=2,
    )

    assert selected == ["videoA", "videoB"]


def test_select_real_val_source_video_ids_validates_requested_ids() -> None:
    rows = [SimpleNamespace(source_video_id="videoA", clip_path="a1.pt")]

    with pytest.raises(ValueError, match="Unknown real selection source videos"):
        select_real_val_source_video_ids(
            rows,
            requested_source_video_ids=["missing"],
            requested_count=1,
        )


def test_normalized_checkpoint_model_config_keeps_real_siglip2_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_physical_move_model,
        "load_transformers_config",
        lambda _model_name: SimpleNamespace(model_type="siglip2"),
    )

    normalized = normalized_checkpoint_model_config(
        {
            "vision_encoder_type": "siglip2",
            "vision_encoder_name": "real-siglip2",
        }
    )

    assert normalized["vision_encoder_type"] == "siglip2"
