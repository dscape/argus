from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_PATH = _PROJECT_ROOT / "study" / "detr-minimal" / "data.py"
_MODEL_PATH = _PROJECT_ROOT / "study" / "detr-minimal" / "model.py"

_DATA_MODULE_NAME = "study_detr_minimal_data"
_data_spec = importlib.util.spec_from_file_location(_DATA_MODULE_NAME, _DATA_PATH)
if _data_spec is None or _data_spec.loader is None:
    raise RuntimeError(f"Failed to load module from {_DATA_PATH}")
detr_data = importlib.util.module_from_spec(_data_spec)
sys.modules[_DATA_MODULE_NAME] = detr_data
old_data_module = sys.modules.get("data")
sys.modules["data"] = detr_data
_data_spec.loader.exec_module(detr_data)
_MODEL_MODULE_NAME = "study_detr_minimal_model"
_model_spec = importlib.util.spec_from_file_location(_MODEL_MODULE_NAME, _MODEL_PATH)
if _model_spec is None or _model_spec.loader is None:
    raise RuntimeError(f"Failed to load module from {_MODEL_PATH}")
detr_model = importlib.util.module_from_spec(_model_spec)
sys.modules[_MODEL_MODULE_NAME] = detr_model
_model_spec.loader.exec_module(detr_model)
if old_data_module is None:
    sys.modules.pop("data", None)
else:
    sys.modules["data"] = old_data_module


def test_hungarian_match_prefers_aligned_type_and_square_queries() -> None:
    num_type_classes = len(detr_data.TYPE_CLASS_NAMES)
    num_square_classes = len(detr_data.SQUARE_OUTPUT_NAMES)
    outputs = {
        "type_logits": torch.full((1, 2, num_type_classes), -10.0),
        "square_logits": torch.full((1, 2, num_square_classes), -10.0),
        "presence_logits": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
    }
    outputs["type_logits"][0, 0, detr_data.TYPE_CLASS_TO_INDEX["R"]] = 10.0
    outputs["square_logits"][0, 0, detr_data.SQUARE_OUTPUT_TO_INDEX["e4"]] = 10.0
    outputs["type_logits"][0, 1, detr_data.TYPE_CLASS_TO_INDEX["q"]] = 10.0
    outputs["square_logits"][0, 1, detr_data.SQUARE_OUTPUT_TO_INDEX["d8"]] = 10.0
    targets = [
        {
            "piece_types": torch.tensor(
                [detr_data.TYPE_CLASS_TO_INDEX["R"], detr_data.TYPE_CLASS_TO_INDEX["q"]],
                dtype=torch.long,
            ),
            "square_indices": torch.tensor(
                [detr_data.SQUARE_OUTPUT_TO_INDEX["e4"], detr_data.SQUARE_OUTPUT_TO_INDEX["d8"]],
                dtype=torch.long,
            ),
        }
    ]
    assignments = detr_model.hungarian_match(
        outputs,
        targets,
        lambda_square=2.0,
        lambda_presence=1.0,
    )
    prediction_indices, target_indices = assignments[0]
    assert prediction_indices.tolist() == [0, 1]
    assert target_indices.tolist() == [0, 1]


def test_decode_predictions_deduplicates_square_and_keeps_hovering_piece() -> None:
    num_type_classes = len(detr_data.TYPE_CLASS_NAMES)
    num_square_classes = len(detr_data.SQUARE_OUTPUT_NAMES)
    outputs = {
        "type_logits": torch.full((1, 3, num_type_classes), -10.0),
        "square_logits": torch.full((1, 3, num_square_classes), -10.0),
        "presence_logits": torch.tensor([[5.0, 4.0, 5.0]], dtype=torch.float32),
    }
    outputs["type_logits"][0, 0, detr_data.TYPE_CLASS_TO_INDEX["R"]] = 10.0
    outputs["square_logits"][0, 0, detr_data.SQUARE_OUTPUT_TO_INDEX["e4"]] = 10.0
    outputs["type_logits"][0, 1, detr_data.TYPE_CLASS_TO_INDEX["N"]] = 10.0
    outputs["square_logits"][0, 1, detr_data.SQUARE_OUTPUT_TO_INDEX["e4"]] = 10.0
    outputs["type_logits"][0, 2, detr_data.TYPE_CLASS_TO_INDEX["q"]] = 10.0
    outputs["square_logits"][0, 2, detr_data.NO_SQUARE_INDEX] = 10.0
    decoded = detr_model.decode_predictions(outputs, presence_threshold=0.5)[0]
    board_index = detr_data.square_name_to_board_index("e4")
    assert decoded["board_labels"][board_index] == detr_data.TYPE_CLASS_TO_INDEX["R"]
    assert decoded["pieces"] == (("R", "e4"), ("q", None))
