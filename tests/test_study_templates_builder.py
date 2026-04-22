from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
from pipeline.shared import fen_to_square_labels
from study.templates.builder.template_bank import (
    STANDARD_START_FEN,
    STARTING_POSITION_PIECE_COUNTS,
    TemplateBankConfig,
    build_template_bank,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _PROJECT_ROOT / "study" / "base-head" / "data.py"
_MODULE_NAME = "study_base_head_data_for_templates_builder_test"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_MODULE_PATH}")
base_head_data = importlib.util.module_from_spec(_SPEC)
__import__("sys").modules[_MODULE_NAME] = base_head_data
_SPEC.loader.exec_module(base_head_data)


def test_build_template_bank_populates_all_piece_types_with_expected_counts(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    labels = tuple(int(label) for row in fen_to_square_labels(STANDARD_START_FEN) for label in row)
    corners = ((0.0, 0.0), (223.0, 0.0), (223.0, 223.0), (0.0, 223.0))
    rows = [
        base_head_data.BoardRow(
            row_id=f"row-{index}",
            image_path=str(image_path),
            corners=corners,
            labels=labels,
        )
        for index in range(2)
    ]
    output_path = tmp_path / "template_bank.pt"
    preview_path = tmp_path / "template_preview.png"

    payload = build_template_bank(
        rows,
        tournament_id="test-tournament",
        output_path=output_path,
        preview_path=preview_path,
        config=TemplateBankConfig(
            jitter_variations=9,
            jitter_pixels=4,
            batch_size=4,
            max_base_crops_per_piece_type=None,
        ),
        embed_fn=lambda crop: torch.tensor(
            [float(crop.mean()), float(crop.std())],
            dtype=torch.float32,
        ),
    )
    loaded = torch.load(output_path, map_location="cpu", weights_only=False)

    assert output_path.exists()
    assert preview_path.exists()
    for piece_type, piece_count in STARTING_POSITION_PIECE_COUNTS.items():
        expected_base_crops = piece_count * len(rows)
        expected_embeddings = expected_base_crops * 9
        assert (
            payload["candidate_base_crop_counts_by_piece_type"][piece_type] == expected_base_crops
        )
        assert payload["base_crop_counts_by_piece_type"][piece_type] == expected_base_crops
        assert payload["embedding_counts_by_piece_type"][piece_type] == expected_embeddings
        assert loaded["embeddings_by_piece_type"][piece_type].shape == (expected_embeddings, 2)


def test_build_template_bank_respects_selection_cap(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    labels = tuple(int(label) for row in fen_to_square_labels(STANDARD_START_FEN) for label in row)
    corners = ((0.0, 0.0), (223.0, 0.0), (223.0, 223.0), (0.0, 223.0))
    rows = [
        base_head_data.BoardRow(
            row_id=f"row-{index}",
            image_path=str(image_path),
            corners=corners,
            labels=labels,
        )
        for index in range(3)
    ]

    payload = build_template_bank(
        rows,
        tournament_id="test-tournament",
        output_path=tmp_path / "template_bank.pt",
        preview_path=tmp_path / "template_preview.png",
        config=TemplateBankConfig(
            jitter_variations=9,
            jitter_pixels=4,
            batch_size=4,
            max_base_crops_per_piece_type=1,
        ),
        embed_fn=lambda crop: torch.tensor([float(crop.mean())], dtype=torch.float32),
    )

    for piece_type in STARTING_POSITION_PIECE_COUNTS:
        assert payload["base_crop_counts_by_piece_type"][piece_type] == 1
        assert payload["embedding_counts_by_piece_type"][piece_type] == 9
