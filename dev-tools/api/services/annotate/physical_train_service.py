"""Service layer for manually labeled non-held-out physical-board training data."""

from __future__ import annotations

from typing import Any

from api.services.annotate import physical_eval_service
from pipeline.physical import eval_dataset, manual_train_dataset


def list_clip_files(
    clips_dir: str = "data/argus/train_real",
    *,
    limit: int = 200,
) -> dict[str, Any]:
    return physical_eval_service._list_clip_files(
        manual_train_dataset,
        clips_dir,
        limit=limit,
        exclude_source_video_ids=set(eval_dataset.get_held_out_source_video_ids()),
    )



def get_annotation_summary() -> dict[str, Any]:
    return physical_eval_service._get_annotation_summary(manual_train_dataset)



def get_frame_annotation(clip_path: str, frame_index: int) -> dict[str, Any] | None:
    return physical_eval_service._get_frame_annotation(
        manual_train_dataset,
        clip_path,
        frame_index,
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



def save_annotation(
    session_id: str,
    clip_path: str,
    frame_index: int,
    corners: list[list[float]],
    labels: list[int | None],
    *,
    output_size: int = manual_train_dataset.DEFAULT_BOARD_SIZE,
) -> dict[str, Any]:
    return physical_eval_service._save_annotation(
        manual_train_dataset,
        session_id,
        clip_path,
        frame_index,
        corners,
        labels,
        output_size=output_size,
    )
