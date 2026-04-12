from pathlib import Path

import torch

from argus.model.argus_model import ArgusModel
from argus.model.patch_pooling import PatchPoolingHead
from argus.model.vision_encoder import VisionEncoder


def test_argus_model_config_round_trip_small() -> None:
    model = ArgusModel(
        vision_encoder_name="facebook/dinov2-small",
        vision_embed_dim=384,
        frozen_vision=True,
        temporal_d_model=256,
        temporal_n_layers=4,
        temporal_d_state=64,
        temporal_expand=2,
        move_vocab_size=1970,
        pooling_type="square_attention",
        square_pool_size=8,
        square_head_enabled=True,
        square_vocab_size=13,
        use_detector=False,
    )

    rebuilt = ArgusModel.from_config(model.model_config)

    assert rebuilt.model_config == model.model_config


def test_argus_model_config_round_trip_yolo() -> None:
    model = ArgusModel(
        vision_encoder_type="yolo",
        vision_encoder_name="weights/yolo_base/yolo11n.pt",
        vision_embed_dim=None,
        frozen_vision=True,
        temporal_d_model=256,
        temporal_n_layers=4,
        temporal_d_state=64,
        temporal_expand=2,
        move_vocab_size=1970,
        pooling_type="mean",
        square_pool_size=8,
        square_head_enabled=False,
        use_detector=False,
        vision_feature_layer_indices=[16, 19, 22],
        vision_output_grid_size=14,
    )

    rebuilt = ArgusModel.from_config(model.model_config)

    assert rebuilt.model_config == model.model_config
    assert model.model_config["vision_embed_dim"] == 448


def test_yolo_vision_encoder_returns_square_token_grid() -> None:
    weights = Path("weights/yolo_base/yolo11n.pt")
    assert weights.exists()

    encoder = VisionEncoder(
        encoder_type="yolo",
        model_name=str(weights),
        frozen=True,
        feature_layer_indices=[16, 19, 22],
        output_grid_size=14,
    )

    frames = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    tokens = encoder.forward_patches(frames)

    assert tokens.shape == (1, 14 * 14, 448)


def test_patch_pooling_head_mean_matches_manual_mean() -> None:
    tokens = torch.arange(2 * 17 * 4, dtype=torch.float32).reshape(2, 17, 4)
    head = PatchPoolingHead(embed_dim=4, pooling_type="mean")

    pooled = head(tokens)

    assert torch.allclose(pooled, tokens[:, 1:, :].mean(dim=1))


def test_patch_pooling_head_square_attention_returns_embed_dim() -> None:
    tokens = torch.randn(3, 257, 8)
    head = PatchPoolingHead(embed_dim=8, pooling_type="square_attention", square_size=8)

    pooled = head(tokens)

    assert pooled.shape == (3, 8)
