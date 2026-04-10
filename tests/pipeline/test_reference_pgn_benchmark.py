"""Tests for reference PGN benchmarking of generated clips."""

from __future__ import annotations

from pathlib import Path

import torch
from pipeline.overlay.reference_pgn_benchmark import benchmark_reference_game


def _write_pgn(path: Path, *, white: str, black: str, result: str, moves: str) -> None:
    path.write_text(
        "\n".join(
            [
                f'[White "{white}"]',
                f'[Black "{black}"]',
                f'[Result "{result}"]',
                "",
                f"{moves} {result}",
                "",
            ]
        )
    )


def _write_clip(path: Path, move_ucis: list[str], *, start: float) -> None:
    torch.save(
        {
            "move_ucis": move_ucis,
            "segment_start_time_seconds": start,
            "segment_end_time_seconds": start + len(move_ucis),
        },
        path,
    )


def test_benchmark_reference_game_reports_exact_coverage_and_gaps(tmp_path: Path) -> None:
    pgn_path = tmp_path / "demo123.pgn"
    _write_pgn(
        pgn_path,
        white="White",
        black="Black",
        result="1-0",
        moves="1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6",
    )

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "clip_overlay_demo123_0.pt", ["e2e4", "e7e5"], start=5.0)
    _write_clip(
        clips_dir / "clip_overlay_demo123_1.pt",
        ["f1b5", "a7a6", "b5a4"],
        start=15.0,
    )
    _write_clip(
        clips_dir / "clip_overlay_demo123_2.pt",
        ["g1f3", "c7c6"],
        start=10.0,
    )

    result = benchmark_reference_game(pgn_path, video_id="demo123", clips_dir=clips_dir)

    assert result["reference_plies"] == 8
    assert result["coverage_plies"] == 5
    assert result["coverage_runs"] == [
        {"start_ply": 0, "end_ply": 1, "plies": 2},
        {"start_ply": 4, "end_ply": 6, "plies": 3},
    ]
    assert result["gaps"] == [
        {"start_ply": 2, "end_ply": 3, "plies": 2},
        {"start_ply": 7, "end_ply": 7, "plies": 1},
    ]

    clips = {Path(clip["clip_path"]).name: clip for clip in result["clips"]}
    assert clips["clip_overlay_demo123_0.pt"]["exact_match_offsets"] == [0]
    assert clips["clip_overlay_demo123_1.pt"]["exact_match_offsets"] == [4]
    assert clips["clip_overlay_demo123_2.pt"]["exact_match_offsets"] == []
    assert clips["clip_overlay_demo123_2.pt"]["longest_prefix_start_ply"] == 2
    assert clips["clip_overlay_demo123_2.pt"]["longest_prefix_plies"] == 1
