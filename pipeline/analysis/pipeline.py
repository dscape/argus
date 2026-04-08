"""Shared local video analysis pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.analysis.board_reading import build_frame_reader
from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.analysis.frame_extractor import extract_frames, sample_frames
from pipeline.analysis.video_annotator import annotate_video
from pipeline.overlay.overlay_move_detector import GameSegment, detect_moves

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of analyzing a local chess video."""

    pgn_files: list[Path] = field(default_factory=list)
    pgn_strings: list[str] = field(default_factory=list)
    segments: list[GameSegment] = field(default_factory=list)
    annotated_video: Path | None = None
    scene_description: str = ""
    total_moves: int = 0


class VideoAnalysisPipeline:
    """Analyze a local chess video into PGN and optional annotations."""

    def __init__(self, config: VideoAnalysisConfig | None = None) -> None:
        self.config = config or VideoAnalysisConfig()

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> AnalysisResult:
        video_path = Path(video_path)
        out_dir = Path(output_dir) if output_dir else self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        result = AnalysisResult()
        scene_context = self._run_scene_analysis(video_path)
        if scene_context is not None:
            result.scene_description = scene_context.description

        fen_sequence = self._run_detection(video_path)
        result.segments = self._resolve_moves(fen_sequence)
        result.total_moves = sum(segment.num_moves for segment in result.segments)

        white = "?"
        black = "?"
        if scene_context is not None:
            white = scene_context.players.get("white", "?")
            black = scene_context.players.get("black", "?")

        result.pgn_strings, result.pgn_files = self._write_pgn(
            result.segments,
            out_dir,
            video_path.stem,
            white=white,
            black=black,
        )

        if self.config.annotate and result.total_moves > 0:
            result.annotated_video = self._annotate(video_path, result.segments, out_dir)

        logger.info(
            "Analysis complete: %d moves across %d game(s)",
            result.total_moves,
            len(result.segments),
        )
        return result

    def _run_scene_analysis(self, video_path: Path) -> Any | None:
        if self.config.scene_backend == "none":
            return None
        if self.config.scene_backend != "vlm":
            raise ValueError(f"Unsupported scene backend: {self.config.scene_backend}")

        from pipeline.analysis.vlm import analyze_scene

        logger.info("=== Scene Analysis ===")
        frames = sample_frames(video_path, count=self.config.vlm_sample_count)
        if not frames:
            logger.warning("No frames extracted for scene analysis")
            return None

        context = analyze_scene([frame.image for frame in frames], self.config)
        logger.info(
            "Scene: type=%s overlay=%s players=%s",
            context.scene_type,
            context.has_overlay,
            context.players,
        )
        return context

    def _run_detection(self, video_path: Path) -> list[tuple[int, float, str | None]]:
        logger.info("=== Board Reading ===")
        reader = build_frame_reader(self.config)
        fen_sequence: list[tuple[int, float, str | None]] = []
        detected = 0
        total = 0

        for frame_data in extract_frames(video_path, fps=self.config.fps):
            total += 1
            read_result = reader.read(frame_data.image)
            fen_sequence.append((frame_data.index, frame_data.timestamp, read_result.fen))
            if read_result.fen is not None:
                detected += 1

            if total % 50 == 0:
                logger.info(
                    "  Processed %d frames, %d readable (%.0f%%)",
                    total,
                    detected,
                    100 * detected / total if total else 0,
                )

        logger.info("Detection complete: %d/%d readable frames", detected, total)
        return fen_sequence

    def _resolve_moves(
        self,
        fen_sequence: list[tuple[int, float, str | None]],
    ) -> list[GameSegment]:
        if not fen_sequence:
            return []

        fens: list[str | None] = []
        frame_indices: list[int] = []
        for frame_idx, _timestamp, fen in fen_sequence:
            fens.append(fen)
            frame_indices.append(frame_idx)

        return detect_moves(
            fens=fens,
            frame_indices=frame_indices,
            fps=self.config.fps,
            stability_window=self.config.stability_window,
        )

    def _write_pgn(
        self,
        segments: list[GameSegment],
        output_dir: Path,
        video_stem: str,
        white: str,
        black: str,
    ) -> tuple[list[str], list[Path]]:
        from argus.chess.pgn_writer import PGNWriter
        from argus.types import MoveEvent

        pgn_strings: list[str] = []
        pgn_files: list[Path] = []

        for segment_index, segment in enumerate(segments):
            if not segment.moves:
                continue

            events = [
                MoveEvent(
                    board_id=0,
                    move_uci=move.move_uci,
                    fen_before=move.fen_before,
                    fen_after=move.fen_after,
                    confidence=move.confidence,
                    frame_idx=move.frame_idx,
                )
                for move in segment.moves
            ]
            pgn = PGNWriter.from_move_events(
                events=events,
                white=white,
                black=black,
                event="Argus Video Analysis",
            )
            suffix = f"_game{segment_index}" if len(segments) > 1 else ""
            pgn_path = output_dir / f"{video_stem}{suffix}.pgn"
            pgn_path.write_text(pgn)
            pgn_strings.append(pgn)
            pgn_files.append(pgn_path)

        return pgn_strings, pgn_files

    def _annotate(
        self,
        video_path: Path,
        segments: list[GameSegment],
        output_dir: Path,
    ) -> Path:
        logger.info("=== Video Annotation ===")
        output_path = output_dir / f"{video_path.stem}_annotated.mp4"
        return annotate_video(video_path, segments, output_path, self.config)
