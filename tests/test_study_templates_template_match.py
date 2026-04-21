from __future__ import annotations

import cv2
import numpy as np
from study.templates.inference.embedder import DEFAULT_ENCODER_TYPE, embed
from study.templates.inference.template_match import classify_crop


def test_template_match_classifier_picks_the_right_piece_type_with_high_confidence() -> None:
    pawn_crop = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(pawn_crop, (112, 156), 30, (220, 220, 220), thickness=-1)
    cv2.rectangle(pawn_crop, (92, 48), (132, 156), (180, 180, 180), thickness=-1)

    knight_crop = np.zeros((224, 224, 3), dtype=np.uint8)
    knight_points = np.array([[64, 176], [152, 176], [124, 48], [84, 88]], dtype=np.int32)
    cv2.fillConvexPoly(knight_crop, knight_points, (220, 220, 220))

    pawn_embedding = embed(pawn_crop, encoder_type=DEFAULT_ENCODER_TYPE, device="cpu")
    knight_embedding = embed(knight_crop, encoder_type=DEFAULT_ENCODER_TYPE, device="cpu")

    template_bank = {
        "encoder_config": {
            "encoder_type": DEFAULT_ENCODER_TYPE,
            "input_size": 224,
            "device": "cpu",
        },
        "embeddings_by_piece_type": {
            "P": pawn_embedding.unsqueeze(0),
            "N": knight_embedding.unsqueeze(0),
        },
    }

    pawn_result = classify_crop(pawn_crop, template_bank)
    knight_result = classify_crop(knight_crop, template_bank)

    assert pawn_result.piece_type == "P"
    assert pawn_result.confidence > 0.999
    assert knight_result.piece_type == "N"
    assert knight_result.confidence > 0.999
