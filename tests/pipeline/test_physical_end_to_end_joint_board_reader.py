from __future__ import annotations

from pathlib import Path

import torch
from pipeline.physical.end_to_end_joint_board_reader import (
    EndToEndJointBoardReaderConfig,
    EndToEndPhysicalJointBoardReader,
)

from argus.model.vision_encoder import VisionEncoder


def test_end_to_end_joint_board_reader_forward_shapes() -> None:
    weights = Path("weights/yolo_base/yolo11n.pt")
    assert weights.exists()

    encoder = VisionEncoder(
        encoder_type="yolo",
        model_name=str(weights),
        frozen=True,
        feature_layer_indices=[16, 19, 22],
        output_grid_size=14,
    )
    model = EndToEndPhysicalJointBoardReader(
        vision_encoder=encoder,
        config=EndToEndJointBoardReaderConfig(
            input_size=224,
            num_classes=13,
            square_query_num_heads=8,
            head_type="pos_mlp",
            hidden_dim=64,
            transformer_layers=1,
            transformer_heads=8,
            transformer_ff_dim=128,
            dropout=0.0,
        ),
    )
    images = torch.zeros(2, 3, 224, 224, dtype=torch.float32)
    corners = torch.tensor(
        [
            [[0.0, 0.0], [223.0, 0.0], [223.0, 223.0], [0.0, 223.0]],
            [[12.0, 8.0], [210.0, 16.0], [220.0, 220.0], [10.0, 208.0]],
        ],
        dtype=torch.float32,
    )

    logits = model(images, corners)

    assert logits.shape == (2, 64, 13)
