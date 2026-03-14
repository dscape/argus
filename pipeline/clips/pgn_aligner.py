"""Align detected moves from video to PGN move sequence."""

import io
import logging
from dataclasses import dataclass

import chess
import chess.pgn

logger = logging.getLogger(__name__)


@dataclass
class AlignedMove:
    """A move aligned between PGN and video detection."""
    move_index: int          # Index in the PGN (0-based half-moves)
    move_uci: str            # UCI notation
    move_san: str            # SAN notation
    frame_idx: int           # Video frame index
    timestamp_seconds: float
    fen_before: str          # Board state before move
    fen_after: str           # Board state after move
    detection_score: float   # Frame diff score at this point
    is_error: bool = False   # Detected motion doesn't match expected move


@dataclass
class AlignmentResult:
    """Result of aligning PGN to detected moves."""
    moves: list[AlignedMove]
    total_pgn_moves: int
    total_detected_moves: int
    aligned_count: int
    error_count: int
    quality: float          # 0.0-1.0 alignment quality score


def align_pgn_to_detections(
    pgn_moves: str,
    detected_moves: list[dict],
    fps: float,
) -> AlignmentResult:
    """Align PGN moves to detected move timestamps.

    Strategy: Walk through PGN moves and detected moves in order,
    matching each PGN move to the next detected motion event.

    Args:
        pgn_moves: PGN movetext string.
        detected_moves: List of dicts with 'frame_idx', 'timestamp_seconds', 'score'.
        fps: Video frames per second.

    Returns:
        AlignmentResult with aligned moves and quality metrics.
    """
    # Parse PGN
    game = chess.pgn.read_game(io.StringIO(pgn_moves))
    if game is None:
        return AlignmentResult(
            moves=[], total_pgn_moves=0,
            total_detected_moves=len(detected_moves),
            aligned_count=0, error_count=0, quality=0.0,
        )

    board = game.board()
    pgn_move_list = []
    node = game
    while node.variations:
        node = node.variations[0]
        pgn_move_list.append({
            "uci": node.move.uci(),
            "san": board.san(node.move),
            "fen_before": board.fen(),
        })
        board.push(node.move)
        pgn_move_list[-1]["fen_after"] = board.fen()

    total_pgn = len(pgn_move_list)
    total_detected = len(detected_moves)

    if total_pgn == 0 or total_detected == 0:
        return AlignmentResult(
            moves=[], total_pgn_moves=total_pgn,
            total_detected_moves=total_detected,
            aligned_count=0, error_count=0, quality=0.0,
        )

    # Greedy alignment: match each PGN move to the next detection
    aligned = []
    det_idx = 0
    errors = 0

    for pgn_idx, pgn_move in enumerate(pgn_move_list):
        if det_idx >= total_detected:
            # Ran out of detections — remaining moves unaligned
            break

        detection = detected_moves[det_idx]
        det_idx += 1

        # Check if this detection makes sense for the expected move
        # (We can't do per-square validation without the warp, so we
        # trust the sequential alignment and flag low-score detections)
        is_error = detection["score"] < 0.003  # Very low motion

        if is_error:
            errors += 1

        aligned.append(AlignedMove(
            move_index=pgn_idx,
            move_uci=pgn_move["uci"],
            move_san=pgn_move["san"],
            frame_idx=detection["frame_idx"],
            timestamp_seconds=detection["timestamp_seconds"],
            fen_before=pgn_move["fen_before"],
            fen_after=pgn_move["fen_after"],
            detection_score=detection["score"],
            is_error=is_error,
        ))

    aligned_count = len(aligned)
    quality = aligned_count / total_pgn if total_pgn > 0 else 0.0

    # Penalize for errors
    if aligned_count > 0:
        quality *= (1.0 - errors / aligned_count * 0.5)

    return AlignmentResult(
        moves=aligned,
        total_pgn_moves=total_pgn,
        total_detected_moves=total_detected,
        aligned_count=aligned_count,
        error_count=errors,
        quality=quality,
    )
