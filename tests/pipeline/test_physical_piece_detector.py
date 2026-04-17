from __future__ import annotations

import torch
import torch.nn as nn
from pipeline.physical.piece_detector import (
    PieceDetector,
    PieceDetectorConfig,
    predict_detections,
)


class _StubPatchEncoder(nn.Module):
    """Emits (batch, n_patches, embed_dim) with fixed geometry."""

    embed_dim = 8

    def __init__(self, *, grid_size: int = 4) -> None:
        super().__init__()
        self.grid_size = grid_size

    def forward_patches(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        return torch.zeros(
            (batch, self.grid_size * self.grid_size, self.embed_dim), device=images.device
        )


def test_piece_detector_forward_shapes() -> None:
    encoder = _StubPatchEncoder(grid_size=4)
    detector = PieceDetector(
        vision_encoder=encoder,
        config=PieceDetectorConfig(
            num_classes=12, num_queries=32, num_decoder_layers=2, num_heads=2, image_size=256
        ),
    )

    outputs = detector(torch.zeros((2, 3, 256, 256)))

    assert outputs["class_logits"].shape == (2, 32, 13)
    assert outputs["bbox_normalized"].shape == (2, 32, 4)
    assert float(outputs["bbox_normalized"].min().item()) >= 0.0
    assert float(outputs["bbox_normalized"].max().item()) <= 1.0


def test_predict_detections_filters_low_score_queries() -> None:
    encoder = _StubPatchEncoder(grid_size=4)
    detector = PieceDetector(
        vision_encoder=encoder,
        config=PieceDetectorConfig(
            num_classes=12, num_queries=16, num_decoder_layers=1, num_heads=2, image_size=256
        ),
    )
    with torch.no_grad():
        detector.class_head.weight.zero_()
        detector.class_head.bias.zero_()
        detector.class_head.bias[-1] = 10.0  # no-object

    detections_batch = predict_detections(
        detector, torch.zeros((1, 3, 256, 256)), score_threshold=0.5
    )

    assert detections_batch == [[]]


def test_predict_detections_maps_bbox_to_pixel_space() -> None:
    encoder = _StubPatchEncoder(grid_size=4)
    detector = PieceDetector(
        vision_encoder=encoder,
        config=PieceDetectorConfig(
            num_classes=12, num_queries=4, num_decoder_layers=1, num_heads=2, image_size=256
        ),
    )
    with torch.no_grad():
        detector.class_head.weight.zero_()
        detector.class_head.bias.zero_()
        detector.class_head.bias[3] = 10.0
        for module in detector.bbox_head:
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                module.bias.zero_()

    detections_batch = predict_detections(
        detector, torch.zeros((1, 3, 256, 256)), score_threshold=0.0
    )

    assert len(detections_batch) == 1
    assert len(detections_batch[0]) == 4
    for detection in detections_batch[0]:
        assert detection.piece_label == 3
        # sigmoid(0) = 0.5 -> (cx, cy, w, h) = (0.5, 0.5, 0.5, 0.5)
        # -> bbox (0.25 * 256 = 64, 64, 192, 192)
        assert abs(detection.xmin - 64.0) < 1.0
        assert abs(detection.ymin - 64.0) < 1.0
        assert abs(detection.xmax - 192.0) < 1.0
        assert abs(detection.ymax - 192.0) < 1.0
