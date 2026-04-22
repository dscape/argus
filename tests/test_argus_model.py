from pathlib import Path

import pytest
import torch

import argus.model.vision_encoder as vision_encoder_module
from argus.model.argus_model import ArgusModel
from argus.model.oblique_square_decoder import ObliqueSquareQueryDecoder
from argus.model.patch_pooling import PatchPoolingHead
from argus.model.vision_encoder import (
    DEFAULT_SIGLIP2_MODEL,
    DEFAULT_SIGLIP_MODEL,
    VisionEncoder,
    default_model_name_for_encoder_type,
)


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


def test_dino_vision_encoder_supports_feature_layer_averaging() -> None:
    encoder = VisionEncoder(
        encoder_type="dinov2",
        model_name="facebook/dinov2-small",
        frozen=True,
        feature_layer_indices=[8, 10, 11],
    )

    frames = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    tokens = encoder.forward_patches(frames)

    assert tokens.shape == (1, 257, 384)


def test_default_model_name_for_encoder_type_matches_expected_checkpoints() -> None:
    assert default_model_name_for_encoder_type("dinov2") == "facebook/dinov2-base"
    assert default_model_name_for_encoder_type("siglip") == DEFAULT_SIGLIP_MODEL
    assert default_model_name_for_encoder_type("siglip2") == DEFAULT_SIGLIP2_MODEL
    assert default_model_name_for_encoder_type("yolo") == "weights/yolo_base/yolo11n.pt"


def test_siglip2_backbone_rejects_siglip_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyConfig:
        model_type = "siglip"

    monkeypatch.setattr(
        vision_encoder_module,
        "_load_transformers_config",
        lambda _path: DummyConfig(),
    )

    with pytest.raises(ValueError, match="encoder_type='siglip2'"):
        vision_encoder_module.Siglip2Backbone(model_name="broken-siglip2")


def test_siglip2_backbone_loads_true_siglip2_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyConfig:
        model_type = "siglip2"

    class DummyVisionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type(
                "DummyVisionConfig",
                (),
                {"hidden_size": 32, "num_hidden_layers": 2, "patch_size": 16},
            )()
            self.encoder = type(
                "DummyEncoder",
                (),
                {"layers": [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]},
            )()

    class DummySiglip2Model:
        def __init__(self) -> None:
            self.vision_model = DummyVisionModel()

        @staticmethod
        def from_pretrained(*_args, **_kwargs) -> "DummySiglip2Model":
            return DummySiglip2Model()

    monkeypatch.setattr(
        vision_encoder_module,
        "_load_transformers_config",
        lambda _path: DummyConfig(),
    )
    monkeypatch.setattr(vision_encoder_module, "Siglip2Model", DummySiglip2Model)

    backbone = vision_encoder_module.Siglip2Backbone(model_name="real-siglip2")

    assert backbone.embed_dim == 32
    assert backbone.patch_size == 16


def test_patch_pooling_head_mean_matches_manual_mean() -> None:
    tokens = torch.arange(2 * 17 * 4, dtype=torch.float32).reshape(2, 17, 4)
    head = PatchPoolingHead(embed_dim=4, pooling_type="mean")

    pooled = head(tokens)

    assert torch.allclose(pooled, tokens[:, 1:, :].mean(dim=1))


def test_patch_pooling_head_square_attention_returns_embed_dim() -> None:
    tokens = torch.randn(3, 257, 8)
    head = PatchPoolingHead(embed_dim=8, pooling_type="square_attention", square_size=8)

    pooled = head(tokens)
    square_tokens = head.to_square_tokens(tokens)
    pooled_from_square_tokens = head.pool_square_tokens(square_tokens)

    assert pooled.shape == (3, 8)
    assert pooled_from_square_tokens.shape == (3, 8)


def test_oblique_square_query_decoder_returns_64_tokens() -> None:
    decoder = ObliqueSquareQueryDecoder(embed_dim=16, num_heads=4)
    patch_tokens = torch.randn(2, 14 * 14, 16)
    corners = torch.tensor(
        [
            [[0.0, 0.0], [223.0, 0.0], [223.0, 223.0], [0.0, 223.0]],
            [[8.0, 12.0], [210.0, 18.0], [220.0, 216.0], [16.0, 204.0]],
        ],
        dtype=torch.float32,
    )

    square_tokens = decoder(patch_tokens, corners=corners, image_size=224)

    assert square_tokens.shape == (2, 64, 16)


def test_argus_model_forward_supports_oblique_square_queries() -> None:
    weights = Path("weights/yolo_base/yolo11n.pt")
    assert weights.exists()

    model = ArgusModel(
        vision_encoder_type="yolo",
        vision_encoder_name=str(weights),
        vision_embed_dim=None,
        frozen_vision=True,
        temporal_d_model=128,
        temporal_n_layers=1,
        temporal_d_state=32,
        temporal_expand=2,
        move_vocab_size=1970,
        pooling_type="square_attention",
        square_pool_size=8,
        square_head_enabled=True,
        square_head_type="pos_mlp",
        square_head_hidden_dim=64,
        square_head_dropout=0.0,
        square_token_mode="oblique_square_queries",
        square_query_num_heads=8,
        use_detector=False,
        vision_feature_layer_indices=[16, 19, 22],
        vision_output_grid_size=14,
    )
    crops = torch.zeros(1, 2, 3, 224, 224, dtype=torch.float32)
    board_corners = torch.tensor(
        [
            [
                [[0.0, 0.0], [223.0, 0.0], [223.0, 223.0], [0.0, 223.0]],
                [[10.0, 6.0], [214.0, 14.0], [220.0, 220.0], [8.0, 210.0]],
            ]
        ],
        dtype=torch.float32,
    )
    legal_masks = torch.ones(1, 2, 1970, dtype=torch.bool)

    output = model(crops=crops, board_corners=board_corners, legal_masks=legal_masks)

    assert output.square_logits is not None
    assert output.square_logits.shape == (1, 2, 64, 13)
    assert output.move_logits.shape == (1, 2, 1, 1970)
