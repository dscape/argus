"""Full Argus model assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from argus.model.board_detector import BoardDetector
from argus.model.move_head import MoveHead
from argus.model.temporal import MambaTemporalModule
from argus.model.vision_encoder import VisionEncoder
from argus.types import ModelOutput


class ArgusModel(nn.Module):
    """End-to-end model for multi-board chess move prediction from video."""

    def __init__(
        self,
        vision_encoder_name: str = "facebook/dinov2-base",
        vision_embed_dim: int = 768,
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
        use_detector: bool = False,
    ) -> None:
        super().__init__()
        self.use_detector = use_detector
        self.vision_encoder = VisionEncoder(
            model_name=vision_encoder_name, frozen=frozen_vision, embed_dim=vision_embed_dim,
        )
        self.temporal = MambaTemporalModule(
            d_model=temporal_d_model, n_layers=temporal_n_layers,
            d_state=temporal_d_state, expand=temporal_expand, input_dim=vision_embed_dim,
        )
        self.move_head = MoveHead(hidden_dim=temporal_d_model, vocab_size=move_vocab_size)
        if use_detector:
            self.board_detector = BoardDetector(
                num_queries=num_board_queries, hidden_dim=detector_hidden_dim,
                num_heads=detector_num_heads, num_decoder_layers=detector_num_layers,
                identity_dim=identity_dim, input_dim=vision_embed_dim,
            )

    def forward_single_board(
        self, crops: torch.Tensor, legal_masks: torch.Tensor | None = None,
    ) -> ModelOutput:
        B, T, C, H, W = crops.shape
        flat_crops = crops.reshape(B * T, C, H, W)
        features = self.vision_encoder.forward_pooled(flat_crops)
        features = features.reshape(B, T, -1)
        temporal_features = self.temporal(features)
        move_logits, move_probs, detect_logits = self.move_head(temporal_features, legal_masks)
        return ModelOutput(
            move_logits=move_logits.unsqueeze(2),
            move_probs=move_probs.unsqueeze(2),
            detect_logits=detect_logits.unsqueeze(2),
        )

    def forward_multi_board(
        self, frames: torch.Tensor, board_crops: torch.Tensor | None = None,
        board_ids: torch.Tensor | None = None, legal_masks: torch.Tensor | None = None,
    ) -> ModelOutput:
        B, T, C, H, W = frames.shape
        flat_frames = frames.reshape(B * T, C, H, W)
        patch_features = self.vision_encoder.forward_patches(flat_frames)
        bboxes, confidences, identities = self.board_detector(patch_features)
        NQ = bboxes.shape[1]
        bboxes = bboxes.reshape(B, T, NQ, 4)
        confidences = confidences.reshape(B, T, NQ)
        identities = identities.reshape(B, T, NQ, -1)

        if board_crops is not None:
            N = board_crops.shape[2]
            flat_crops = board_crops.reshape(B * T * N, C, H, W)
            crop_features = self.vision_encoder.forward_pooled(flat_crops)
            crop_features = crop_features.reshape(B, T, N, -1)
            all_move_logits, all_move_probs, all_detect_logits = [], [], []
            for board_idx in range(N):
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
            move_logits=move_logits, move_probs=move_probs, detect_logits=detect_logits,
            board_bboxes=bboxes, board_confidence=confidences, board_identity=identities,
        )

    def forward(
        self, crops: torch.Tensor | None = None, frames: torch.Tensor | None = None,
        legal_masks: torch.Tensor | None = None, board_crops: torch.Tensor | None = None,
        board_ids: torch.Tensor | None = None,
    ) -> ModelOutput:
        if crops is not None:
            return self.forward_single_board(crops, legal_masks)
        elif frames is not None and self.use_detector:
            return self.forward_multi_board(frames, board_crops, board_ids, legal_masks)
        else:
            raise ValueError("Provide 'crops' (single-board) or 'frames' with use_detector=True")
