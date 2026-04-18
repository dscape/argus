"""Helpers for manually labeled physical-board training data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pipeline.physical.shared import annotation_dataset, splits

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_SPLIT = "train"
DATASET_ROOT = _PROJECT_ROOT / "data" / "physical" / DATASET_SPLIT
BOARDS_DIR = DATASET_ROOT / "boards"
SQUARES_DIR = DATASET_ROOT / "squares"
BOARD_ANNOTATIONS_PATH = DATASET_ROOT / "board_annotations.jsonl"
SQUARE_MANIFEST_PATH = DATASET_ROOT / "square_manifest.jsonl"
TRANSIENT_ANNOTATIONS_PATH = DATASET_ROOT / "transient_annotations.jsonl"
DEFAULT_BOARD_SIZE = annotation_dataset.DEFAULT_BOARD_SIZE


SavedBoardAnnotation = annotation_dataset.SavedBoardAnnotation
SavedSquareCrop = annotation_dataset.SavedSquareCrop
SavedTransientAnnotation = annotation_dataset.SavedTransientAnnotation
SavedTransientMoveAnnotation = annotation_dataset.SavedTransientMoveAnnotation
SavedHandOcclusionSpan = annotation_dataset.SavedHandOcclusionSpan
rectify_board_image = annotation_dataset.rectify_board_image
extract_square_crops = annotation_dataset.extract_square_crops


def load_board_annotation(clip_path: str, frame_index: int) -> dict[str, Any] | None:
    splits.ensure_annotation_layout_migrated()
    return annotation_dataset.load_board_annotation(
        BOARD_ANNOTATIONS_PATH,
        clip_path=clip_path,
        frame_index=frame_index,
    )


def list_board_annotations(clip_path: str) -> list[dict[str, Any]]:
    splits.ensure_annotation_layout_migrated()
    return annotation_dataset.list_board_annotations(BOARD_ANNOTATIONS_PATH, clip_path=clip_path)


def delete_board_annotation(clip_path: str, frame_index: int) -> bool:
    splits.ensure_annotation_layout_migrated()
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
    image_corners: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
    corner_space: str = "clip_frame",
    clip_frame_size: list[int] | tuple[int, int] | None = None,
    native_corners: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
    native_image_bbox: list[int] | tuple[int, int, int, int] | None = None,
    source_frame_index: int | None = None,
) -> dict[str, Any]:
    splits.ensure_annotation_layout_migrated()
    splits.assign_source_video_split(source_video_id, DATASET_SPLIT)
    return annotation_dataset.save_board_annotation(
        _PROJECT_ROOT,
        dataset_root=DATASET_ROOT,
        boards_dir=BOARDS_DIR,
        squares_dir=SQUARES_DIR,
        board_annotations_path=BOARD_ANNOTATIONS_PATH,
        square_manifest_path=SQUARE_MANIFEST_PATH,
        split=DATASET_SPLIT,
        image_rgb=image_rgb,
        clip_path=clip_path,
        frame_index=frame_index,
        source_video_id=source_video_id,
        corners=corners,
        labels=labels,
        output_size=output_size,
        image_corners=image_corners,
        corner_space=corner_space,
        clip_frame_size=clip_frame_size,
        native_corners=native_corners,
        native_image_bbox=native_image_bbox,
        source_frame_index=source_frame_index,
    )


def load_transient_annotation(clip_path: str) -> dict[str, Any] | None:
    splits.ensure_annotation_layout_migrated()
    return annotation_dataset.load_transient_annotation(
        TRANSIENT_ANNOTATIONS_PATH,
        clip_path=clip_path,
    )


def save_transient_annotation(
    *,
    clip_path: str,
    source_video_id: str | None,
    move_annotations: list[dict[str, Any]],
    hand_occlusion_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    splits.ensure_annotation_layout_migrated()
    splits.assign_source_video_split(source_video_id, DATASET_SPLIT)
    return annotation_dataset.save_transient_annotation(
        TRANSIENT_ANNOTATIONS_PATH,
        clip_path=clip_path,
        source_video_id=source_video_id,
        move_annotations=move_annotations,
        hand_occlusion_spans=hand_occlusion_spans,
    )


def delete_transient_annotation(clip_path: str) -> bool:
    splits.ensure_annotation_layout_migrated()
    return annotation_dataset.delete_transient_annotation(
        TRANSIENT_ANNOTATIONS_PATH,
        clip_path=clip_path,
    )


def get_saved_frame_counts_by_clip() -> dict[str, int]:
    splits.ensure_annotation_layout_migrated()
    return annotation_dataset.get_saved_frame_counts_by_clip(BOARD_ANNOTATIONS_PATH)


def get_source_video_ids() -> list[str]:
    return splits.get_source_video_ids_for_split(DATASET_SPLIT)


def get_annotation_summary() -> dict[str, Any]:
    splits.ensure_annotation_layout_migrated()
    return annotation_dataset.get_annotation_summary(
        _PROJECT_ROOT,
        dataset_root=DATASET_ROOT,
        board_annotations_path=BOARD_ANNOTATIONS_PATH,
        square_manifest_path=SQUARE_MANIFEST_PATH,
    )
