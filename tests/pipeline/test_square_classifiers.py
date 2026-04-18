from __future__ import annotations

import torch
import torch.nn as nn
from pipeline.physical.two_stage.classifiers import (
    OCCUPANCY_CLASS_NAMES,
    OCCUPANCY_NUM_CLASSES,
    PIECE_CLASS_NAMES,
    PIECE_NUM_CLASSES,
    SquareClassifier,
    SquareClassifierConfig,
    build_occupancy_classifier,
    build_piece_classifier,
    piece_label_to_square_class,
    square_class_to_occupancy_label,
    square_class_to_piece_label,
)


class _StubEncoder(nn.Module):
    embed_dim = 8

    def forward_pooled(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        return torch.zeros((batch, self.embed_dim), device=images.device)


def test_square_class_to_occupancy_label_maps_empty_to_zero_and_others_to_one() -> None:
    assert square_class_to_occupancy_label(0) == 0
    for class_id in range(1, 13):
        assert square_class_to_occupancy_label(class_id) == 1


def test_square_class_to_piece_label_roundtrips_with_piece_label_to_square_class() -> None:
    for class_id in range(1, 13):
        piece_label = square_class_to_piece_label(class_id)
        assert 0 <= piece_label < PIECE_NUM_CLASSES
        assert piece_label_to_square_class(piece_label) == class_id


def test_occupancy_classifier_forward_produces_two_logits() -> None:
    encoder = _StubEncoder()
    model = build_occupancy_classifier(encoder)

    logits = model(torch.zeros((3, 3, 96, 96)))

    assert logits.shape == (3, OCCUPANCY_NUM_CLASSES)


def test_piece_classifier_forward_produces_twelve_logits() -> None:
    encoder = _StubEncoder()
    model = build_piece_classifier(encoder)

    logits = model(torch.zeros((2, 3, 192, 192)))

    assert logits.shape == (2, PIECE_NUM_CLASSES)
    assert len(PIECE_CLASS_NAMES) == PIECE_NUM_CLASSES
    assert len(OCCUPANCY_CLASS_NAMES) == OCCUPANCY_NUM_CLASSES


def test_square_classifier_checkpoint_config_roundtrips() -> None:
    encoder = _StubEncoder()
    config = SquareClassifierConfig(num_classes=12, dropout=0.25)
    model = SquareClassifier(vision_encoder=encoder, config=config)

    serialized = model.checkpoint_config()

    assert serialized == {"classifier_config": {"num_classes": 12, "dropout": 0.25}}


def test_square_classifier_rejects_num_classes_below_two() -> None:
    try:
        SquareClassifierConfig(num_classes=1)
    except ValueError as error:
        assert "num_classes" in str(error)
    else:
        raise AssertionError("expected ValueError for num_classes < 2")
