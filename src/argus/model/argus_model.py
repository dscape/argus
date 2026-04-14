"""Full Argus model assembly."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from argus.model.board_detector import BoardDetector
from argus.model.move_head import MoveHead
from argus.model.oblique_square_decoder import ObliqueSquareQueryDecoder
from argus.model.patch_pooling import PatchPoolingHead
from argus.model.square_head import SquareHead
from argus.model.temporal import MambaTemporalModule
from argus.model.vision_encoder import VisionEncoder
from argus.types import ModelOutput


class ArgusModel(nn.Module):
    """End-to-end model for multi-board chess move prediction from video."""

    def __init__(
        self,
        vision_encoder_name: str = "facebook/dinov2-base",
        vision_embed_dim: int | None = 768,
        frozen_vision: bool = True,
        temporal_d_model: int = 512,
        temporal_n_layers: int = 6,
        temporal_d_state: int = 128,
        temporal_expand: int = 2,
        move_vocab_size: int = 1970,
        num_board_queries: int = 32,
        detector_hidden_dim: int = 512,
        detector_num_heads: int = 8,
        detector_num_layers: int = 6,
        identity_dim: int = 128,
        pooling_type: str = "mean",
        square_pool_size: int = 8,
        square_head_enabled: bool = False,
        square_vocab_size: int = 13,
        square_token_mode: str = "pooled",
        square_query_num_heads: int = 8,
        square_query_dropout: float = 0.0,
        square_query_mlp_ratio: float = 4.0,
        use_detector: bool = False,
        vision_encoder_type: str = "dinov2",
        vision_feature_layer_indices: list[int] | tuple[int, ...] | None = None,
        vision_output_grid_size: int = 14,
    ) -> None:
        super().__init__()
        self.use_detector = use_detector
        self.square_token_mode = square_token_mode
        self.vision_encoder = VisionEncoder(
            model_name=vision_encoder_name,
            frozen=frozen_vision,
            embed_dim=vision_embed_dim,
            encoder_type=vision_encoder_type,
            feature_layer_indices=vision_feature_layer_indices,
            output_grid_size=vision_output_grid_size,
        )
        resolved_embed_dim = self.vision_encoder.embed_dim
        feature_layers = None
        if vision_feature_layer_indices is not None:
            feature_layers = [int(idx) for idx in vision_feature_layer_indices]
        self._model_config = {
            "vision_encoder_name": vision_encoder_name,
            "vision_encoder_type": vision_encoder_type,
            "vision_embed_dim": resolved_embed_dim,
            "vision_feature_layer_indices": feature_layers,
            "vision_output_grid_size": vision_output_grid_size,
            "frozen_vision": frozen_vision,
            "temporal_d_model": temporal_d_model,
            "temporal_n_layers": temporal_n_layers,
            "temporal_d_state": temporal_d_state,
            "temporal_expand": temporal_expand,
            "move_vocab_size": move_vocab_size,
            "num_board_queries": num_board_queries,
            "detector_hidden_dim": detector_hidden_dim,
            "detector_num_heads": detector_num_heads,
            "detector_num_layers": detector_num_layers,
            "identity_dim": identity_dim,
            "pooling_type": pooling_type,
            "square_pool_size": square_pool_size,
            "square_head_enabled": square_head_enabled,
            "square_vocab_size": square_vocab_size,
            "square_token_mode": square_token_mode,
            "square_query_num_heads": square_query_num_heads,
            "square_query_dropout": square_query_dropout,
            "square_query_mlp_ratio": square_query_mlp_ratio,
            "use_detector": use_detector,
        }
        self.patch_pooling = PatchPoolingHead(
            embed_dim=resolved_embed_dim,
            pooling_type=pooling_type,
            square_size=square_pool_size,
        )
        self.square_tokenizer = self._build_square_tokenizer(
            embed_dim=resolved_embed_dim,
            square_token_mode=square_token_mode,
            square_query_num_heads=square_query_num_heads,
            square_query_dropout=square_query_dropout,
            square_query_mlp_ratio=square_query_mlp_ratio,
        )
        self.temporal = MambaTemporalModule(
            d_model=temporal_d_model,
            n_layers=temporal_n_layers,
            d_state=temporal_d_state,
            expand=temporal_expand,
            input_dim=resolved_embed_dim,
        )
        self.move_head = MoveHead(hidden_dim=temporal_d_model, vocab_size=move_vocab_size)
        self.square_head = (
            SquareHead(embed_dim=resolved_embed_dim, num_classes=square_vocab_size)
            if square_head_enabled
            else None
        )
        if use_detector:
            self.board_detector = BoardDetector(
                num_queries=num_board_queries,
                hidden_dim=detector_hidden_dim,
                num_heads=detector_num_heads,
                num_decoder_layers=detector_num_layers,
                identity_dim=identity_dim,
                input_dim=resolved_embed_dim,
            )

    def _build_square_tokenizer(
        self,
        *,
        embed_dim: int,
        square_token_mode: str,
        square_query_num_heads: int,
        square_query_dropout: float,
        square_query_mlp_ratio: float,
    ) -> ObliqueSquareQueryDecoder | None:
        if square_token_mode == "pooled":
            return None
        if square_token_mode == "oblique_square_queries":
            return ObliqueSquareQueryDecoder(
                embed_dim=embed_dim,
                num_heads=square_query_num_heads,
                dropout=square_query_dropout,
                mlp_ratio=square_query_mlp_ratio,
            )
        raise ValueError(f"Unsupported square_token_mode: {square_token_mode}")

    def _square_tokens_from_patches(
        self,
        patch_tokens: torch.Tensor,
        *,
        board_corners: torch.Tensor | None,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor | None:
        if self.square_tokenizer is not None:
            square_tokens: torch.Tensor = self.square_tokenizer(
                patch_tokens,
                corners=board_corners,
                image_size=(image_height, image_width),
            )
            return square_tokens
        if self.square_head is not None or self.patch_pooling.pooling_type == "square_attention":
            square_tokens = self.patch_pooling.to_square_tokens(patch_tokens)
            return square_tokens
        return None

    def forward_single_board(
        self,
        crops: torch.Tensor,
        legal_masks: torch.Tensor | None = None,
        board_corners: torch.Tensor | None = None,
    ) -> ModelOutput:
        batch_size, seq_len, _channels, image_height, image_width = crops.shape
        flat_crops = crops.reshape(batch_size * seq_len, crops.shape[2], image_height, image_width)
        patch_tokens = self.vision_encoder.forward_patches(flat_crops)
        flat_board_corners = None
        if board_corners is not None:
            flat_board_corners = board_corners.reshape(batch_size * seq_len, 4, 2)

        square_tokens = self._square_tokens_from_patches(
            patch_tokens,
            board_corners=flat_board_corners,
            image_height=image_height,
            image_width=image_width,
        )
        square_logits = None
        if self.square_head is not None:
            if square_tokens is None:
                raise RuntimeError("Square head requires square tokens, but none were produced")
            square_logits = self.square_head(square_tokens).reshape(
                batch_size,
                seq_len,
                -1,
                self._model_config["square_vocab_size"],
            )

        if square_tokens is not None and self.square_tokenizer is not None:
            features = self.patch_pooling.pool_square_tokens(square_tokens)
        else:
            features = self.patch_pooling(patch_tokens)
        features = features.reshape(batch_size, seq_len, -1)
        temporal_features = self.temporal(features)
        move_logits, move_probs, detect_logits = self.move_head(temporal_features, legal_masks)
        return ModelOutput(
            move_logits=move_logits.unsqueeze(2),
            move_probs=move_probs.unsqueeze(2),
            detect_logits=detect_logits.unsqueeze(2),
            square_logits=square_logits,
        )

    def forward_multi_board(
        self,
        frames: torch.Tensor,
        board_crops: torch.Tensor | None = None,
        board_ids: torch.Tensor | None = None,
        legal_masks: torch.Tensor | None = None,
    ) -> ModelOutput:
        batch_size, seq_len, channels, image_height, image_width = frames.shape
        flat_frames = frames.reshape(batch_size * seq_len, channels, image_height, image_width)
        patch_features = self.vision_encoder.forward_patches(flat_frames)
        bboxes, confidences, identities = self.board_detector(patch_features)
        num_queries = bboxes.shape[1]
        bboxes = bboxes.reshape(batch_size, seq_len, num_queries, 4)
        confidences = confidences.reshape(batch_size, seq_len, num_queries)
        identities = identities.reshape(batch_size, seq_len, num_queries, -1)

        if board_crops is not None:
            num_boards = board_crops.shape[2]
            flat_crops = board_crops.reshape(
                batch_size * seq_len * num_boards, channels, image_height, image_width
            )
            patch_tokens = self.vision_encoder.forward_patches(flat_crops)
            crop_features = self.patch_pooling(patch_tokens)
            crop_features = crop_features.reshape(batch_size, seq_len, num_boards, -1)
            all_move_logits, all_move_probs, all_detect_logits = [], [], []
            for board_idx in range(num_boards):
                board_features = crop_features[:, :, board_idx, :]
                temporal_out = self.temporal(board_features)
                board_masks = legal_masks[:, :, board_idx, :] if legal_masks is not None else None
                m_logits, m_probs, d_logits = self.move_head(temporal_out, board_masks)
                all_move_logits.append(m_logits)
                all_move_probs.append(m_probs)
                all_detect_logits.append(d_logits)
            move_logits = torch.stack(all_move_logits, dim=2)
            move_probs = torch.stack(all_move_probs, dim=2)
            detect_logits = torch.stack(all_detect_logits, dim=2)
        else:
            raise NotImplementedError("ROI pooling from detected bboxes not yet implemented.")

        return ModelOutput(
            move_logits=move_logits,
            move_probs=move_probs,
            detect_logits=detect_logits,
            board_bboxes=bboxes,
            board_confidence=confidences,
            board_identity=identities,
        )

    @property
    def model_config(self) -> dict[str, Any]:
        return dict(self._model_config)

    @classmethod
    def from_config(cls, model_config: dict[str, Any] | None = None) -> ArgusModel:
        return cls(**(model_config or {}))

    def forward(
        self,
        crops: torch.Tensor | None = None,
        frames: torch.Tensor | None = None,
        legal_masks: torch.Tensor | None = None,
        board_crops: torch.Tensor | None = None,
        board_ids: torch.Tensor | None = None,
        board_corners: torch.Tensor | None = None,
    ) -> ModelOutput:
        if crops is not None:
            return self.forward_single_board(crops, legal_masks, board_corners=board_corners)
        if frames is not None and self.use_detector:
            return self.forward_multi_board(frames, board_crops, board_ids, legal_masks)
        raise ValueError("Provide 'crops' (single-board) or 'frames' with use_detector=True")
