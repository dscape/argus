from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _PROJECT_ROOT / "study" / "base-head" / "data.py"
_MODULE_NAME = "study_base_head_data"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_MODULE_PATH}")
base_head_data = importlib.util.module_from_spec(_SPEC)
sys_modules = __import__("sys").modules
sys_modules[_MODULE_NAME] = base_head_data
_SPEC.loader.exec_module(base_head_data)


def test_square_name_roundtrip() -> None:
    for square_name in ("a1", "d4", "h8"):
        index = base_head_data.square_name_to_index(square_name)
        assert base_head_data.index_to_square_name(index) == square_name


def test_derive_square_targets_from_bboxes_marks_piece_body_when_overlap_is_large() -> None:
    labels = [0] * 64
    labels[0] = 4
    projected_bboxes = np.zeros((64, 4), dtype=np.float64)
    projected_bboxes[0] = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    projected_bboxes[8] = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float64)
    type_labels, base_labels = base_head_data.derive_square_targets_from_bboxes(
        labels=tuple(labels),
        projected_bboxes=projected_bboxes,
        body_overlap_threshold=0.4,
    )
    assert type_labels[0] == 4
    assert base_labels[0] == 1.0
    assert type_labels[8] == base_head_data.PIECE_BODY_TYPE_INDEX
    assert base_labels[8] == 0.0


def test_infer_square_labels_requires_base_and_concrete_piece() -> None:
    type_logits = torch.tensor(
        [
            [0.0, 2.0, 0.0] + [0.0] * 11,
            [0.0, 0.0, 0.0] + [0.0] * 10 + [3.0],
            [3.0, 2.0, 0.0] + [0.0] * 11,
        ],
        dtype=torch.float32,
    )
    base_logits = torch.tensor([[2.0], [2.0], [-2.0]], dtype=torch.float32)
    predicted = base_head_data.infer_square_labels(type_logits, base_logits)
    assert predicted.tolist() == [1, 0, 0]


def test_narrow_bottom_width_bbox_clamps_width_to_square_bottom_edge() -> None:
    corners = ((0.0, 0.0), (80.0, 10.0), (70.0, 80.0), (10.0, 70.0))
    projected_bbox = base_head_data.project_piece_bbox_for_crop_mode(
        corners,
        frame_shape=(96, 96, 3),
        row=4,
        col=3,
        piece_height=2.0,
        crop_mode=base_head_data.PROJECTED_CROP_MODE,
    )
    narrow_bbox = base_head_data.project_piece_bbox_for_crop_mode(
        corners,
        frame_shape=(96, 96, 3),
        row=4,
        col=3,
        piece_height=2.0,
        crop_mode=base_head_data.NARROW_BOTTOM_WIDTH_CROP_MODE,
    )
    bottom_x1, bottom_x2 = base_head_data.square_bottom_edge_x_bounds(corners, row=4, col=3)
    assert narrow_bbox[0] == bottom_x1
    assert narrow_bbox[2] == bottom_x2
    assert narrow_bbox[1] == projected_bbox[1]
    assert narrow_bbox[3] == projected_bbox[3]
    assert (narrow_bbox[2] - narrow_bbox[0]) < (projected_bbox[2] - projected_bbox[0])
