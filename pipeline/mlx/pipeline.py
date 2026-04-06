"""End-to-end MLX chess analysis pipeline orchestrator.

Chains together all stages:
1. Frame extraction
2. VLM scene understanding (Gemma 4)
3. Board segmentation (SAM 3)
4. Piece detection (RF-DETR + PieceClassifier)
5. Move resolution (FEN-diffing)
6. Video annotation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.mlx.config import MLXPipelineConfig
from pipeline.overlay.overlay_move_detector import GameSegment

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of the full MLX chess analysis pipeline."""

    pgn_files: list[Path] = field(default_factory=list)
    pgn_strings: list[str] = field(default_factory=list)
    segments: list[GameSegment] = field(default_factory=list)
    annotated_video: Path | None = None
    scene_description: str = ""
    total_moves: int = 0


class MLXChessPipeline:
    """Orchestrates the full MLX chess video analysis pipeline."""

    def __init__(self, config: MLXPipelineConfig | None = None) -> None:
        self.config = config or MLXPipelineConfig()

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        vlm_only: bool = False,
        skip_vlm: bool = False,
    ) -> AnalysisResult:
        """Run the full analysis pipeline on a video.

        Args:
            video_path: Path to the input video file.
            output_dir: Output directory (defaults to config.output_dir).
            vlm_only: Only run VLM scene analysis, skip detection.
            skip_vlm: Skip VLM, go straight to board detection.

        Returns:
            AnalysisResult with detected moves, PGN, and annotated video.
        """
        video_path = Path(video_path)
        out_dir = Path(output_dir) if output_dir else self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        result = AnalysisResult()

        # ── Stage 1: VLM Scene Understanding ──────────────────────────
        if not skip_vlm:
            result.scene_description = self._run_vlm(video_path)
            if vlm_only:
                return result

        # ── Stage 2-4: Frame-by-frame detection ───────────────────────
        fen_sequence = self._run_detection(video_path)

        # ── Stage 5: Move resolution ─────────────────────────────────
        result.segments = self._resolve_moves(fen_sequence)
        result.total_moves = sum(s.num_moves for s in result.segments)

        # ── Stage 5b: Generate PGN ───────────────────────────────────
        result.pgn_strings, result.pgn_files = self._write_pgn(
            result.segments, out_dir, video_path.stem
        )

        # ── Stage 6: Video annotation ────────────────────────────────
        if self.config.annotate and result.total_moves > 0:
            result.annotated_video = self._annotate(
                video_path,
                result.segments,
                out_dir,
                result.scene_description,
            )

        # ── Summary ──────────────────────────────────────────────────
        logger.info("Analysis complete:")
        logger.info("  Moves detected: %d", result.total_moves)
        logger.info("  Game segments: %d", len(result.segments))
        for i, pgn_path in enumerate(result.pgn_files):
            logger.info("  PGN %d: %s", i, pgn_path)
        if result.annotated_video:
            logger.info("  Annotated video: %s", result.annotated_video)

        return result

    def _run_vlm(self, video_path: Path) -> str:
        """Stage 1: VLM scene understanding."""
        from pipeline.mlx.frame_extractor import sample_frames
        from pipeline.mlx.vlm_analyzer import analyze_scene

        logger.info("=== Stage 1: VLM Scene Analysis ===")

        frames = sample_frames(video_path, count=self.config.vlm_sample_count)
        if not frames:
            logger.warning("No frames extracted for VLM analysis")
            return ""

        context = analyze_scene([f.image for f in frames], self.config)

        print(f"\n{'=' * 60}")
        print("Scene Analysis (Gemma 4)")
        print(f"{'=' * 60}")
        print(f"Type: {context.scene_type}")
        print(f"Overlay: {context.has_overlay}")
        print(f"Players: {context.players}")
        print(f"Phase: {context.game_phase}")
        print(f"Notes: {context.additional_notes}")
        print(f"{'=' * 60}\n")

        return context.description

    def _run_detection(
        self, video_path: Path
    ) -> list[tuple[int, float, str | None]]:
        """Stages 2-4: Board detection + piece classification per frame.

        Locks onto the grid after the first stable detection so that
        frame-to-frame noise in grid alignment doesn't corrupt FENs.
        """
        import cv2

        from pipeline.mlx.frame_extractor import extract_frames
        from pipeline.overlay.grid_detector import find_board_in_frame
        from pipeline.overlay.piece_classifier import read_fen_with_grid

        logger.info("=== Stage 2-4: Board Detection ===")

        fen_sequence: list[tuple[int, float, str | None]] = []
        detected = 0
        total = 0

        # Grid locking: overlay boards don't move, so once we find a
        # grid that produces a plausible FEN (has both kings, 8 ranks),
        # reuse it for every subsequent frame.
        locked_grid = None

        for frame_data in extract_frames(video_path, fps=self.config.fps):
            total += 1
            frame_bgr = cv2.cvtColor(frame_data.image, cv2.COLOR_RGB2BGR)

            fen: str | None = None

            if locked_grid is not None:
                # Fast path: reuse locked grid
                fen = read_fen_with_grid(frame_bgr, locked_grid, device="mps")
            else:
                # Discovery phase: find grid and validate FEN quality
                grid = find_board_in_frame(frame_bgr)
                if grid is None:
                    fen_sequence.append((frame_data.index, frame_data.timestamp, None))
                    continue

                fen = read_fen_with_grid(frame_bgr, grid, device="mps")

                # Validate: a real board has exactly 8 ranks and both kings
                has_both_kings = "K" in fen and "k" in fen
                if has_both_kings:
                    locked_grid = grid
                    logger.info(
                        "Grid locked: sq=%d, h0=%d, v0=%d (frame %d)",
                        grid.sq_size,
                        grid.h_lines[0],
                        grid.v_lines[0],
                        frame_data.index,
                    )
                else:
                    # Pre-lock FEN without both kings is garbage — skip
                    fen_sequence.append((frame_data.index, frame_data.timestamp, None))
                    continue

            fen_sequence.append((frame_data.index, frame_data.timestamp, fen))
            detected += 1

            if total % 50 == 0:
                logger.info(
                    "  Processed %d frames, %d with valid FEN (%.0f%%)",
                    total,
                    detected,
                    100 * detected / total if total else 0,
                )

        logger.info(
            "Detection complete: %d/%d frames with valid board state",
            detected,
            total,
        )
        return fen_sequence

    def _resolve_moves(
        self, fen_sequence: list[tuple[int, float, str | None]]
    ) -> list[GameSegment]:
        """Stage 5: Resolve FEN sequence into moves."""
        from pipeline.mlx.move_resolver import resolve_moves

        logger.info("=== Stage 5: Move Resolution ===")
        return resolve_moves(fen_sequence, self.config)

    def _write_pgn(
        self,
        segments: list[GameSegment],
        output_dir: Path,
        video_stem: str,
    ) -> tuple[list[str], list[Path]]:
        """Generate PGN files from game segments."""
        from pipeline.mlx.move_resolver import segments_to_pgn

        pgn_strings = segments_to_pgn(segments, event="MLX Analysis")
        pgn_files: list[Path] = []

        for i, pgn in enumerate(pgn_strings):
            suffix = f"_game{i}" if len(pgn_strings) > 1 else ""
            pgn_path = output_dir / f"{video_stem}{suffix}.pgn"
            pgn_path.write_text(pgn)
            pgn_files.append(pgn_path)
            print(f"\n{'=' * 60}")
            print(f"Game {i + 1} PGN:")
            print(f"{'=' * 60}")
            print(pgn)

        return pgn_strings, pgn_files

    def _annotate(
        self,
        video_path: Path,
        segments: list[GameSegment],
        output_dir: Path,
        game_info: str,
    ) -> Path:
        """Stage 6: Annotate video with detected moves."""
        from pipeline.mlx.video_annotator import annotate_video

        logger.info("=== Stage 6: Video Annotation ===")

        output_path = output_dir / f"{video_path.stem}_annotated.mp4"
        return annotate_video(
            video_path,
            segments,
            output_path,
            self.config,
            game_info=game_info,
        )
