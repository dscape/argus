"""Audit real-video clip generation without writing output clips."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pipeline.overlay.overlay_clip_generator import ClipGenerationDiagnostics, generate_from_video

_FAILURE_PRIORITY = [
    "clip_build_rejected",
    "illegal_jump_fragmentation",
    "repeated_hard_cuts",
    "midgame_pickup_no_long_span",
    "segments_too_short",
    "too_few_readable_frames",
    "no_game_segments_detected",
    "no_saved_clips",
]


def audit_video_generation(
    *,
    video_id: str,
    video_path: str,
    channel_handle: str,
    output_dir: str = "data/argus/train_real",
    base_fps: float = 2.0,
    min_moves_per_segment: int = 5,
) -> dict[str, Any]:
    """Run clip generation in dry-run mode and summarize the outcome."""
    diagnostics: list[ClipGenerationDiagnostics] = []
    results = generate_from_video(
        video_path,
        channel_handle=channel_handle,
        output_dir=output_dir,
        base_fps=base_fps,
        min_moves_per_segment=min_moves_per_segment,
        save_clips=False,
        diagnostics=diagnostics,
    )
    return summarize_video_audit(
        video_id=video_id,
        diagnostics=diagnostics,
        generated_results=results,
        min_moves_per_segment=min_moves_per_segment,
    )


def summarize_video_audit(
    *,
    video_id: str,
    diagnostics: list[ClipGenerationDiagnostics],
    generated_results: list[dict[str, Any]],
    min_moves_per_segment: int,
) -> dict[str, Any]:
    """Summarize dry-run clip generation diagnostics for one source video."""
    clip_reports = [
        _summarize_clip_audit(report, min_moves_per_segment=min_moves_per_segment)
        for report in diagnostics
    ]

    failure_counter = Counter(
        report["failure_reason"] for report in clip_reports if report["failure_reason"]
    )
    dominant_failure = None if generated_results else _pick_dominant_failure(failure_counter)

    all_segment_moves = [
        moves
        for report in diagnostics
        for moves in report.segment_move_counts
    ]
    all_saved_moves = [
        moves
        for report in diagnostics
        for moves in report.saved_clip_move_counts
    ]
    all_short_moves = [
        moves
        for report in diagnostics
        for moves in report.short_segment_move_counts
    ]
    all_rejected_moves = [
        moves
        for report in diagnostics
        for moves in report.rejected_segment_move_counts
    ]

    hard_cut_count = sum(
        report.move_detection.hard_cut_count
        for report in diagnostics
        if report.move_detection is not None
    )
    illegal_jump_count = sum(
        report.move_detection.illegal_jump_count
        for report in diagnostics
        if report.move_detection is not None
    )
    started_midgame = any(
        report.move_detection is not None and report.move_detection.started_midgame
        for report in diagnostics
    )

    return {
        "video_id": video_id,
        "generated_clip_count": len(generated_results),
        "failure_reason": dominant_failure,
        "sampled_frame_count": sum(report.sampled_frame_count for report in diagnostics),
        "readable_fen_count": sum(report.readable_fen_count for report in diagnostics),
        "segment_count": len(all_segment_moves),
        "max_segment_moves": max(all_segment_moves, default=0),
        "saved_clip_move_counts": all_saved_moves,
        "short_segment_move_counts": all_short_moves,
        "rejected_segment_move_counts": all_rejected_moves,
        "hard_cut_count": hard_cut_count,
        "illegal_jump_count": illegal_jump_count,
        "started_midgame": started_midgame,
        "clip_reports": clip_reports,
    }


def _summarize_clip_audit(
    diagnostics: ClipGenerationDiagnostics,
    *,
    min_moves_per_segment: int,
) -> dict[str, Any]:
    max_segment_moves = max(diagnostics.segment_move_counts, default=0)
    hard_cut_count = diagnostics.move_detection.hard_cut_count if diagnostics.move_detection else 0
    illegal_jump_count = (
        diagnostics.move_detection.illegal_jump_count if diagnostics.move_detection else 0
    )
    started_midgame = (
        diagnostics.move_detection.started_midgame if diagnostics.move_detection else False
    )

    failure_reason = _classify_clip_audit(
        generated_clip_count=len(diagnostics.saved_clip_move_counts),
        sampled_frame_count=diagnostics.sampled_frame_count,
        readable_fen_count=diagnostics.readable_fen_count,
        segment_count=len(diagnostics.segment_move_counts),
        max_segment_moves=max_segment_moves,
        rejected_segment_count=len(diagnostics.rejected_segment_move_counts),
        hard_cut_count=hard_cut_count,
        illegal_jump_count=illegal_jump_count,
        started_midgame=started_midgame,
        min_moves_per_segment=min_moves_per_segment,
    )

    return {
        "clip_label": diagnostics.clip_label,
        "start_time_seconds": diagnostics.start_time_seconds,
        "end_time_seconds": diagnostics.end_time_seconds,
        "sampled_frame_count": diagnostics.sampled_frame_count,
        "readable_fen_count": diagnostics.readable_fen_count,
        "segment_count": len(diagnostics.segment_move_counts),
        "segment_move_counts": list(diagnostics.segment_move_counts),
        "max_segment_moves": max_segment_moves,
        "saved_clip_move_counts": list(diagnostics.saved_clip_move_counts),
        "short_segment_move_counts": list(diagnostics.short_segment_move_counts),
        "rejected_segment_move_counts": list(diagnostics.rejected_segment_move_counts),
        "hard_cut_count": hard_cut_count,
        "illegal_jump_count": illegal_jump_count,
        "started_midgame": started_midgame,
        "failure_reason": failure_reason,
    }


def _classify_clip_audit(
    *,
    generated_clip_count: int,
    sampled_frame_count: int,
    readable_fen_count: int,
    segment_count: int,
    max_segment_moves: int,
    rejected_segment_count: int,
    hard_cut_count: int,
    illegal_jump_count: int,
    started_midgame: bool,
    min_moves_per_segment: int,
) -> str:
    if generated_clip_count > 0:
        return "would_generate_clips"
    minimum_frames = max(10, min_moves_per_segment * 3)
    if sampled_frame_count < minimum_frames or readable_fen_count < minimum_frames:
        return "too_few_readable_frames"
    if rejected_segment_count > 0 and max_segment_moves >= min_moves_per_segment:
        return "clip_build_rejected"
    if hard_cut_count >= 5 and hard_cut_count >= illegal_jump_count:
        return "repeated_hard_cuts"
    if 0 < max_segment_moves < min_moves_per_segment:
        if illegal_jump_count > 0 and illegal_jump_count >= hard_cut_count:
            return "illegal_jump_fragmentation"
        if started_midgame:
            return "midgame_pickup_no_long_span"
        return "segments_too_short"
    if illegal_jump_count > 0 and illegal_jump_count >= hard_cut_count:
        return "illegal_jump_fragmentation"
    if hard_cut_count > 0:
        return "repeated_hard_cuts"
    if started_midgame:
        return "midgame_pickup_no_long_span"
    if segment_count == 0:
        return "no_game_segments_detected"
    return "no_saved_clips"


def _pick_dominant_failure(failure_counter: Counter[str]) -> str | None:
    if not failure_counter:
        return None
    for failure in _FAILURE_PRIORITY:
        if failure in failure_counter:
            return failure
    return failure_counter.most_common(1)[0][0]
