"""Resolve chess moves from a sequence of board states (FENs).

Bridges MLX pipeline output to the existing Argus chess logic:
- GameStateMachine for move validation
- FEN-diffing algorithm from overlay_move_detector.py
- PGN generation via PGNWriter

This module does NOT re-implement move detection — it reuses the proven
algorithm from pipeline/overlay/overlay_move_detector.py.
"""

from __future__ import annotations

import logging

from pipeline.mlx.config import MLXPipelineConfig
from pipeline.overlay.overlay_move_detector import (
    GameSegment,
    detect_moves,
)

logger = logging.getLogger(__name__)


def resolve_moves(
    fen_sequence: list[tuple[int, float, str | None]],
    config: MLXPipelineConfig,
) -> list[GameSegment]:
    """Resolve moves from a sequence of (frame_idx, timestamp, FEN) tuples.

    Delegates to the existing detect_moves() from overlay_move_detector.py,
    which handles:
    - Stability windows (ignoring animation frames)
    - Hard-cut detection (>4 squares changed = new game)
    - Legal move validation via python-chess
    - Multi-game segment tracking

    Args:
        fen_sequence: List of (frame_index, timestamp_seconds, fen_or_none).
        config: Pipeline configuration.

    Returns:
        List of GameSegment, each containing detected moves and PGN.
    """
    if not fen_sequence:
        return []

    # Separate into parallel lists for detect_moves()
    fens: list[str | None] = []
    frame_indices: list[int] = []
    fps = config.fps

    for frame_idx, timestamp, fen in fen_sequence:
        fens.append(fen)
        frame_indices.append(frame_idx)

    segments = detect_moves(
        fens=fens,
        frame_indices=frame_indices,
        fps=fps,
        stability_window=config.stability_window,
    )

    total_moves = sum(s.num_moves for s in segments)
    logger.info(
        "Resolved %d move(s) across %d game segment(s)",
        total_moves,
        len(segments),
    )

    for i, seg in enumerate(segments):
        if seg.moves:
            logger.info(
                "  Segment %d: %d moves, frames %d-%d (%.1fs-%.1fs)",
                i,
                seg.num_moves,
                seg.start_frame,
                seg.end_frame,
                seg.start_time,
                seg.end_time,
            )
            logger.info("    Moves: %s", seg.pgn_moves[:80])

    return segments


def segments_to_pgn(
    segments: list[GameSegment],
    white: str = "?",
    black: str = "?",
    event: str = "MLX Analysis",
) -> list[str]:
    """Convert game segments to PGN strings.

    Args:
        segments: Game segments from resolve_moves().
        white: White player name.
        black: Black player name.
        event: Event name for PGN header.

    Returns:
        List of PGN strings, one per segment.
    """
    from argus.chess.pgn_writer import PGNWriter
    from argus.types import MoveEvent

    pgns: list[str] = []

    for seg in segments:
        if not seg.moves:
            continue

        # Convert OverlayDetectedMove to MoveEvent
        events = [
            MoveEvent(
                board_id=0,
                move_uci=m.move_uci,
                fen_before=m.fen_before,
                fen_after=m.fen_after,
                confidence=m.confidence,
                frame_idx=m.frame_idx,
            )
            for m in seg.moves
        ]

        pgn = PGNWriter.from_move_events(
            events=events,
            white=white,
            black=black,
            event=event,
        )
        pgns.append(pgn)

    return pgns
