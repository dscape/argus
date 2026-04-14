from __future__ import annotations

import json

import numpy as np
import torch
from pipeline.physical.oblique_square_context import (
    PhysicalAnnotatedObliqueSquareContextDataset,
    board_to_image_transform,
    extract_oblique_square_context_crops,
    project_square_quad,
)


def test_project_square_quad_maps_board_grid_into_image_space() -> None:
    corners = ((10.0, 20.0), (90.0, 20.0), (90.0, 100.0), (10.0, 100.0))
    transform = board_to_image_transform(corners)

    quad = project_square_quad(transform, row=0, col=0)

    np.testing.assert_allclose(quad[0], np.array([10.0, 20.0]), atol=1e-4)
    np.testing.assert_allclose(quad[2], np.array([20.0, 30.0]), atol=1e-4)


def test_extract_oblique_square_context_crops_returns_64_crops() -> None:
    image = np.full((128, 128, 3), 127, dtype=np.uint8)
    corners = ((16.0, 16.0), (112.0, 20.0), (108.0, 112.0), (20.0, 108.0))

    crops = extract_oblique_square_context_crops(image, corners)

    assert len(crops) == 64
    assert all(crop.ndim == 3 for crop in crops)
    assert all(crop.shape[0] > 0 and crop.shape[1] > 0 for crop in crops)


def test_physical_annotated_oblique_square_context_dataset_loads_clip_frames(tmp_path) -> None:
    project_root = tmp_path
    clip_path = project_root / "data" / "argus" / "train_real" / "clip_test.pt"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    frame = torch.full((3, 64, 64), 127, dtype=torch.uint8)
    torch.save({"frames": torch.stack([frame], dim=0)}, clip_path)

    annotation_root = project_root / "data" / "physical" / "val"
    annotation_root.mkdir(parents=True, exist_ok=True)
    board_annotations_path = annotation_root / "board_annotations.jsonl"
    row = {
        "annotation_id": "ann-1",
        "clip_path": "data/argus/train_real/clip_test.pt",
        "frame_index": 0,
        "source_video_id": "video-1",
        "corners": [[8.0, 8.0], [56.0, 8.0], [56.0, 56.0], [8.0, 56.0]],
        "labels": [0] * 64,
    }
    board_annotations_path.write_text(json.dumps(row) + "\n")

    import pipeline.physical.oblique_square_context as oblique_square_context

    original_root = oblique_square_context._PROJECT_ROOT
    oblique_square_context._PROJECT_ROOT = project_root
    try:
        dataset = PhysicalAnnotatedObliqueSquareContextDataset(
            annotation_root=annotation_root,
            image_size=32,
        )
        images, labels = dataset[0]
    finally:
        oblique_square_context._PROJECT_ROOT = original_root

    assert images.shape == (64, 3, 32, 32)
    assert labels.shape == (64,)
