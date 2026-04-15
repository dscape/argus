"""Service layer for manually labeled physical-board training data."""

from __future__ import annotations

from typing import Any

from pipeline.physical import manual_train_dataset

from api.services.annotate import physical_eval_service


def list_clip_files(
    clips_dir: str = "data/argus/train_real",
    *,
    limit: int = 200,
) -> dict[str, Any]:
    return physical_eval_service._list_clip_files(
        manual_train_dataset,
        clips_dir,
        limit=limit,
        split_name=manual_train_dataset.DATASET_SPLIT,
    )


def get_annotation_summary() -> dict[str, Any]:
    return physical_eval_service._get_annotation_summary(manual_train_dataset)


def get_frame_annotation(
    clip_path: str,
    frame_index: int,
    *,
    session_id: str | None = None,
    padding_px: int = 0,
) -> dict[str, Any] | None:
    return physical_eval_service._get_frame_annotation(
        manual_train_dataset,
        clip_path,
        frame_index,
        session_id=session_id,
        padding_px=padding_px,
    )


def get_move_corrections(session_id: str, clip_path: str) -> dict[str, Any]:
    return physical_eval_service._get_move_corrections(
        manual_train_dataset,
        session_id,
        clip_path,
    )


def delete_annotation(clip_path: str, frame_index: int) -> dict[str, Any] | None:
    return physical_eval_service._delete_annotation(
        manual_train_dataset,
        clip_path,
        frame_index,
    )


rectify_frame = physical_eval_service.rectify_frame
detect_corners = physical_eval_service.detect_corners
track_corners = physical_eval_service.track_corners


def save_annotation(
    session_id: str,
    clip_path: str,
    frame_index: int,
    corners: list[list[float]],
    labels: list[int | None],
    *,
    output_size: int = manual_train_dataset.DEFAULT_BOARD_SIZE,
    padding_px: int = 0,
) -> dict[str, Any]:
    return physical_eval_service._save_annotation(
        manual_train_dataset,
        session_id,
        clip_path,
        frame_index,
        corners,
        labels,
        output_size=output_size,
        padding_px=padding_px,
    )
