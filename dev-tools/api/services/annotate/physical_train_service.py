"""Service layer for manually labeled physical-board training data."""

from __future__ import annotations

from typing import Any

from pipeline.physical.shared import annotation_coverage, manual_train_dataset

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


def list_clip_priorities(
    clips_dir: str = "data/argus/train_real",
    *,
    limit: int = 200,
) -> dict[str, Any]:
    """Return the same clips as ``list_clip_files`` but ordered by ROI action.

    Channels that need val annotations (unlocks probe) come first, then
    channels that need train annotations, then add_diversity. Clips whose
    channel is already saturated are dropped.
    """
    listing = list_clip_files(clips_dir, limit=max(limit, 1000))
    report = annotation_coverage.compute_coverage()
    action_by_clip: dict[str, str] = {}
    channel_by_clip: dict[str, str] = {}
    rank_by_clip: dict[str, int] = {}
    for rank, row in enumerate(report.channels):
        for path in row.clip_paths:
            action_by_clip[path] = row.roi_action
            channel_by_clip[path] = row.channel
            rank_by_clip[path] = rank

    annotated = {
        annotation_coverage.ROI_NEEDS_VAL: 0,
        annotation_coverage.ROI_NEEDS_TRAIN: 1,
        annotation_coverage.ROI_ADD_DIVERSITY: 2,
    }
    priority_clips: list[dict[str, Any]] = []
    for clip in listing["clips"]:
        clip_path = clip["clip_path"]
        action = action_by_clip.get(clip_path)
        if action is None or action == annotation_coverage.ROI_SATURATED:
            continue
        priority_clips.append(
            {
                **clip,
                "source_channel_handle": channel_by_clip.get(clip_path),
                "roi_action": action,
            }
        )
    priority_clips.sort(
        key=lambda clip: (
            annotated.get(clip["roi_action"], 99),
            rank_by_clip.get(clip["clip_path"], 9999),
            clip["clip_path"],
        )
    )
    return {
        "clips_dir": listing["clips_dir"],
        "clips": priority_clips[:limit],
    }


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


def get_transient_annotation(clip_path: str) -> dict[str, Any] | None:
    return physical_eval_service._get_transient_annotation(
        manual_train_dataset,
        clip_path,
    )


def delete_annotation(clip_path: str, frame_index: int) -> dict[str, Any] | None:
    return physical_eval_service._delete_annotation(
        manual_train_dataset,
        clip_path,
        frame_index,
    )


def delete_transient_annotation(clip_path: str) -> bool:
    return physical_eval_service._delete_transient_annotation(
        manual_train_dataset,
        clip_path,
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


def save_transient_annotation(
    clip_path: str,
    move_annotations: list[dict[str, Any]],
    hand_occlusion_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    return physical_eval_service._save_transient_annotation(
        manual_train_dataset,
        clip_path,
        move_annotations,
        hand_occlusion_spans,
    )
