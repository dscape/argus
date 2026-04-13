"""Helpers for manually labeled non-held-out physical-board training data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pipeline.physical import annotation_dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = _PROJECT_ROOT / "data" / "physical" / "train_manual"
BOARDS_DIR = DATASET_ROOT / "boards"
SQUARES_DIR = DATASET_ROOT / "squares"
BOARD_ANNOTATIONS_PATH = DATASET_ROOT / "board_annotations.jsonl"
SQUARE_MANIFEST_PATH = DATASET_ROOT / "square_manifest.jsonl"
DEFAULT_BOARD_SIZE = annotation_dataset.DEFAULT_BOARD_SIZE


SavedBoardAnnotation = annotation_dataset.SavedBoardAnnotation
SavedSquareCrop = annotation_dataset.SavedSquareCrop
rectify_board_image = annotation_dataset.rectify_board_image
extract_square_crops = annotation_dataset.extract_square_crops


def load_board_annotation(clip_path: str, frame_index: int) -> dict[str, Any] | None:
    return annotation_dataset.load_board_annotation(
        BOARD_ANNOTATIONS_PATH,
        clip_path=clip_path,
        frame_index=frame_index,
    )



def list_board_annotations(clip_path: str) -> list[dict[str, Any]]:
    return annotation_dataset.list_board_annotations(BOARD_ANNOTATIONS_PATH, clip_path=clip_path)



def delete_board_annotation(clip_path: str, frame_index: int) -> bool:
    return annotation_dataset.delete_board_annotation(
        _PROJECT_ROOT,
        boards_dir=BOARDS_DIR,
        squares_dir=SQUARES_DIR,
        board_annotations_path=BOARD_ANNOTATIONS_PATH,
        square_manifest_path=SQUARE_MANIFEST_PATH,
        clip_path=clip_path,
        frame_index=frame_index,
    )



def save_board_annotation(
    image_rgb: np.ndarray,
    *,
    clip_path: str,
    frame_index: int,
    source_video_id: str | None,
    corners: list[list[float]],
    labels: list[int | None],
    output_size: int = DEFAULT_BOARD_SIZE,
) -> dict[str, Any]:
    return annotation_dataset.save_board_annotation(
        _PROJECT_ROOT,
        dataset_root=DATASET_ROOT,
        boards_dir=BOARDS_DIR,
        squares_dir=SQUARES_DIR,
        board_annotations_path=BOARD_ANNOTATIONS_PATH,
        square_manifest_path=SQUARE_MANIFEST_PATH,
        split="train_manual",
        image_rgb=image_rgb,
        clip_path=clip_path,
        frame_index=frame_index,
        source_video_id=source_video_id,
        corners=corners,
        labels=labels,
        output_size=output_size,
    )



def get_saved_frame_counts_by_clip() -> dict[str, int]:
    return annotation_dataset.get_saved_frame_counts_by_clip(BOARD_ANNOTATIONS_PATH)



def get_source_video_ids() -> list[str]:
    return annotation_dataset.get_source_video_ids(BOARD_ANNOTATIONS_PATH)



def get_annotation_summary() -> dict[str, Any]:
    return annotation_dataset.get_annotation_summary(
        _PROJECT_ROOT,
        dataset_root=DATASET_ROOT,
        board_annotations_path=BOARD_ANNOTATIONS_PATH,
        square_manifest_path=SQUARE_MANIFEST_PATH,
    )
