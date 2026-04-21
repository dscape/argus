from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _PROJECT_ROOT / "study" / "geometric-mask" / "mask.py"
_MODULE_NAME = "study_geometric_mask"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_MODULE_PATH}")
geometric_mask = importlib.util.module_from_spec(_SPEC)
sys_modules = __import__("sys").modules
sys_modules[_MODULE_NAME] = geometric_mask
_SPEC.loader.exec_module(geometric_mask)


def test_project_local_mask_to_crop_canvas_bottom_aligns_content() -> None:
    local_mask = np.full((2, 4), 255, dtype=np.uint8)
    canvas = geometric_mask.project_local_mask_to_crop_canvas(
        local_mask,
        output_size=8,
        flip_horizontally=False,
    )
    assert canvas.shape == (8, 8)
    assert np.all(canvas[0:4] == 0)
    assert np.all(canvas[4:8, 0:8] == 255)


def test_build_mask_from_closer_polygons_zeros_occluded_region() -> None:
    target_bbox = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    closer_polygon = np.array([[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]], dtype=np.float32)
    mask = geometric_mask.build_mask_from_closer_polygons(
        target_bbox=target_bbox,
        closer_polygons=[closer_polygon],
        output_size=10,
        flip_horizontally=False,
    )
    assert mask.shape == (10, 10)
    assert mask[5, 5] == 0
    assert mask[9, 0] == 255
