"""Tests for dry-run real-video audit summaries."""

from pipeline.overlay.overlay_clip_generator import ClipGenerationDiagnostics
from pipeline.overlay.overlay_move_detector import MoveDetectionDiagnostics
from pipeline.overlay.real_video_audit import summarize_video_audit


def _diag(
    *,
    clip_label: str,
    sampled_frames: int = 100,
    readable_fens: int = 100,
    started_midgame: bool = False,
    hard_cuts: int = 0,
    illegal_jumps: int = 0,
    segment_moves: list[int] | None = None,
    saved_moves: list[int] | None = None,
    short_moves: list[int] | None = None,
    rejected_moves: list[int] | None = None,
) -> ClipGenerationDiagnostics:
    return ClipGenerationDiagnostics(
        clip_label=clip_label,
        sampled_frame_count=sampled_frames,
        readable_fen_count=readable_fens,
        move_detection=MoveDetectionDiagnostics(
            valid_frame_count=readable_fens,
            started_midgame=started_midgame,
            hard_cut_count=hard_cuts,
            illegal_jump_count=illegal_jumps,
            segment_move_counts=list(segment_moves or []),
        ),
        segment_move_counts=list(segment_moves or []),
        saved_clip_move_counts=list(saved_moves or []),
        short_segment_move_counts=list(short_moves or []),
        rejected_segment_move_counts=list(rejected_moves or []),
    )


def test_summarize_video_audit_reports_would_generate_clips() -> None:
    report = summarize_video_audit(
        video_id="demo123",
        diagnostics=[
            _diag(
                clip_label="demo123_clip1",
                segment_moves=[8],
                saved_moves=[8],
            )
        ],
        generated_results=[{"num_moves": 8}],
        min_moves_per_segment=5,
    )

    assert report["failure_reason"] is None
    assert report["generated_clip_count"] == 1
    assert report["saved_clip_move_counts"] == [8]
    assert report["clip_reports"][0]["failure_reason"] == "would_generate_clips"


def test_summarize_video_audit_classifies_short_windows_as_too_few_frames() -> None:
    report = summarize_video_audit(
        video_id="demo123",
        diagnostics=[
            _diag(
                clip_label="demo123_clip1",
                sampled_frames=12,
                readable_fens=12,
                started_midgame=True,
            )
        ],
        generated_results=[],
        min_moves_per_segment=5,
    )

    assert report["failure_reason"] == "too_few_readable_frames"
    assert report["clip_reports"][0]["failure_reason"] == "too_few_readable_frames"


def test_summarize_video_audit_classifies_midgame_short_spans() -> None:
    report = summarize_video_audit(
        video_id="demo123",
        diagnostics=[
            _diag(
                clip_label="demo123_clip1",
                started_midgame=True,
                segment_moves=[3, 4],
                short_moves=[3, 4],
            )
        ],
        generated_results=[],
        min_moves_per_segment=5,
    )

    assert report["failure_reason"] == "midgame_pickup_no_long_span"
    assert report["max_segment_moves"] == 4
    assert report["clip_reports"][0]["failure_reason"] == "midgame_pickup_no_long_span"


def test_summarize_video_audit_prioritizes_illegal_fragmentation() -> None:
    report = summarize_video_audit(
        video_id="demo123",
        diagnostics=[
            _diag(
                clip_label="demo123_clip1",
                illegal_jumps=6,
                segment_moves=[2, 4],
                short_moves=[2, 4],
            ),
            _diag(
                clip_label="demo123_clip2",
                hard_cuts=7,
                segment_moves=[],
            ),
        ],
        generated_results=[],
        min_moves_per_segment=5,
    )

    assert report["failure_reason"] == "illegal_jump_fragmentation"
    assert report["illegal_jump_count"] == 6
    assert report["hard_cut_count"] == 7
