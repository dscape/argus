"""Compare generated real-video clips against a reference PGN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess.pgn
import torch


@dataclass(frozen=True)
class ClipBenchmarkResult:
    clip_path: Path
    segment_start_time_seconds: float
    segment_end_time_seconds: float
    clip_plies: int
    exact_match_offsets: tuple[int, ...]
    longest_prefix_start_ply: int | None
    longest_prefix_plies: int


def benchmark_reference_game(
    pgn_path: str | Path,
    *,
    video_id: str,
    clips_dir: str | Path = "data/argus/train_real",
) -> dict[str, Any]:
    """Benchmark generated clips for one video against a reference PGN."""
    pgn_file = Path(pgn_path)
    reference_game = _load_reference_game(pgn_file)
    reference_moves = [move.uci() for move in reference_game.mainline_moves()]
    clip_results = _benchmark_clips(reference_moves, video_id=video_id, clips_dir=clips_dir)

    covered_plies = _collect_exact_coverage(clip_results)
    coverage_runs = _ranges_from_sorted_indices(covered_plies)
    gap_ranges = _invert_ranges(coverage_runs, total=len(reference_moves))

    return {
        "video_id": video_id,
        "pgn_path": str(pgn_file),
        "white": reference_game.headers.get("White", "?"),
        "black": reference_game.headers.get("Black", "?"),
        "result": reference_game.headers.get("Result", "*"),
        "reference_plies": len(reference_moves),
        "coverage_plies": len(covered_plies),
        "coverage_ratio": (len(covered_plies) / len(reference_moves)) if reference_moves else 0.0,
        "coverage_runs": [
            {"start_ply": start, "end_ply": end, "plies": end - start + 1}
            for start, end in coverage_runs
        ],
        "gaps": [
            {"start_ply": start, "end_ply": end, "plies": end - start + 1}
            for start, end in gap_ranges
        ],
        "clips": [
            {
                "clip_path": str(result.clip_path),
                "segment_start_time_seconds": result.segment_start_time_seconds,
                "segment_end_time_seconds": result.segment_end_time_seconds,
                "clip_plies": result.clip_plies,
                "exact_match_offsets": list(result.exact_match_offsets),
                "longest_prefix_start_ply": result.longest_prefix_start_ply,
                "longest_prefix_plies": result.longest_prefix_plies,
            }
            for result in clip_results
        ],
    }


def _load_reference_game(pgn_path: Path) -> chess.pgn.Game:
    with pgn_path.open() as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"No PGN game found in {pgn_path}")
    return game


def _benchmark_clips(
    reference_moves: list[str],
    *,
    video_id: str,
    clips_dir: str | Path,
) -> list[ClipBenchmarkResult]:
    clip_paths = sorted(Path(clips_dir).glob(f"clip_overlay_{video_id}*.pt"))
    results: list[ClipBenchmarkResult] = []
    for clip_path in clip_paths:
        clip = torch.load(clip_path, map_location="cpu", weights_only=True)
        clip_moves = [str(move) for move in clip.get("move_ucis", [])]
        exact_offsets = tuple(_find_exact_match_offsets(reference_moves, clip_moves))
        prefix_start, prefix_len = _find_best_prefix_match(reference_moves, clip_moves)
        results.append(
            ClipBenchmarkResult(
                clip_path=clip_path,
                segment_start_time_seconds=float(clip.get("segment_start_time_seconds", 0.0)),
                segment_end_time_seconds=float(clip.get("segment_end_time_seconds", 0.0)),
                clip_plies=len(clip_moves),
                exact_match_offsets=exact_offsets,
                longest_prefix_start_ply=prefix_start,
                longest_prefix_plies=prefix_len,
            )
        )
    results.sort(key=lambda result: (result.segment_start_time_seconds, str(result.clip_path)))
    return results


def _find_exact_match_offsets(reference_moves: list[str], clip_moves: list[str]) -> list[int]:
    if not clip_moves or len(clip_moves) > len(reference_moves):
        return []
    offsets: list[int] = []
    stop = len(reference_moves) - len(clip_moves) + 1
    for start in range(stop):
        if reference_moves[start : start + len(clip_moves)] == clip_moves:
            offsets.append(start)
    return offsets


def _find_best_prefix_match(
    reference_moves: list[str],
    clip_moves: list[str],
) -> tuple[int | None, int]:
    if not clip_moves or not reference_moves:
        return None, 0

    best_start: int | None = None
    best_len = 0
    for start in range(len(reference_moves)):
        match_len = 0
        while (
            start + match_len < len(reference_moves)
            and match_len < len(clip_moves)
            and reference_moves[start + match_len] == clip_moves[match_len]
        ):
            match_len += 1
        if match_len > best_len:
            best_len = match_len
            best_start = start
    return best_start, best_len


def _collect_exact_coverage(results: list[ClipBenchmarkResult]) -> list[int]:
    covered: set[int] = set()
    for result in results:
        if not result.exact_match_offsets:
            continue
        start = result.exact_match_offsets[0]
        covered.update(range(start, start + result.clip_plies))
    return sorted(covered)


def _ranges_from_sorted_indices(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = indices[0]
    end = indices[0]
    for index in indices[1:]:
        if index == end + 1:
            end = index
            continue
        ranges.append((start, end))
        start = end = index
    ranges.append((start, end))
    return ranges


def _invert_ranges(ranges: list[tuple[int, int]], *, total: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if not ranges:
        return [(0, total - 1)]

    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = end + 1
    if cursor < total:
        gaps.append((cursor, total - 1))
    return gaps
