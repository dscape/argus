from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from study.templates.eval import evaluator
from study.templates.inference.template_match import TemplateMatchResult
from study.templates.proposals.common import SquareCropProposal


def test_evaluate_template_matching_writes_expected_metrics_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    eval_labels = tmp_path / "labels.jsonl"
    eval_row = {
        "frame_id": "frame-1",
        "image_path": str(image_path),
        "category": "easy-stationary",
        "corners": [[0.0, 0.0], [31.0, 0.0], [31.0, 31.0], [0.0, 31.0]],
        "pieces": [{"type": "P", "square": "a1"}],
    }
    eval_labels.write_text(json.dumps(eval_row) + "\n")

    monkeypatch.setitem(
        evaluator._PROPOSAL_SOURCES,
        "cuboid",
        lambda _frame: [
            SquareCropProposal(square="a1", crop_bgr=np.zeros((16, 16, 3), dtype=np.uint8))
        ],
    )
    monkeypatch.setattr(
        evaluator,
        "_classify_proposals",
        lambda proposals, *, template_bank, device: [
            TemplateMatchResult(
                piece_type="P",
                confidence=0.95,
                margin=0.5,
                piece_similarities={"P": 0.95},
            )
            for _ in proposals
        ],
    )

    template_bank = {"encoder_config": {"encoder_type": "dinov3", "input_size": 224}}
    metrics = evaluator.evaluate_template_matching(
        template_bank=template_bank,
        proposal_source="cuboid",
        output_dir=tmp_path / "run",
        eval_labels=eval_labels,
        device="cpu",
        match_threshold=0.75,
    )

    assert sorted(metrics.keys()) == ["easy-stationary", "macro", "overall"]
    assert metrics["overall"]["count"] == 1
    assert metrics["overall"]["strict_piece_exact_match"] == 1.0
    assert metrics["overall"]["placed_board_exact_match"] == 1.0
    assert metrics["overall"]["per_square_accuracy"] == 1.0
    assert metrics["macro"]["piece_f1_macro"] == (1.0 / 12.0)

    written_metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert sorted(written_metrics["overall"].keys()) == [
        "count",
        "failure_examples",
        "per_piece_f1",
        "per_square_accuracy",
        "piece_f1_macro",
        "placed_board_exact_match",
        "strict_piece_exact_match",
    ]
