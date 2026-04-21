from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import torch
from study.templates.inference.embedder import DEFAULT_ENCODER_TYPE, embed, get_embedder

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _PROJECT_ROOT / "study" / "base-head" / "data.py"
_MODULE_NAME = "study_base_head_data_for_templates_embedder_test"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_MODULE_PATH}")
base_head_data = importlib.util.module_from_spec(_SPEC)
__import__("sys").modules[_MODULE_NAME] = base_head_data
_SPEC.loader.exec_module(base_head_data)


def test_embed_returns_consistent_dinov3_vector_for_saved_chess_crop(tmp_path: Path) -> None:
    record = next(
        eval_record
        for eval_record in base_head_data.load_eval_records("study/eval/labels.jsonl")
        if any(label > 0 for label in eval_record.placed_labels)
    )
    image_path = base_head_data.resolve_project_path(record.image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None

    square_index = next(index for index, label in enumerate(record.placed_labels) if label > 0)
    crop = base_head_data.extract_study_piece_crop(
        image,
        record.corners,
        row=square_index // 8,
        col=square_index % 8,
        output_size=224,
        flip_left_half=True,
    )
    sample_path = tmp_path / "sample_crop.png"
    assert cv2.imwrite(str(sample_path), crop)

    vector_from_array = embed(crop, encoder_type=DEFAULT_ENCODER_TYPE, device="cpu")
    vector_from_file = embed(sample_path, encoder_type=DEFAULT_ENCODER_TYPE, device="cpu")
    embedder = get_embedder(encoder_type=DEFAULT_ENCODER_TYPE, device="cpu")

    assert vector_from_array.shape == (embedder.embedding_dim,)
    assert vector_from_file.shape == (embedder.embedding_dim,)
    assert embedder.embedding_dim == 768
    assert torch.allclose(vector_from_array, vector_from_file, atol=1e-6, rtol=1e-6)
