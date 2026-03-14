"""Post-processing for inference output.

Handles confidence gating, game completion detection,
and PGN cleanup.
"""

from __future__ import annotations

import chess

from argus.chess.state_machine import GameStateMachine
from argus.types import GameTrack, MoveEvent


def confidence_gate_moves(
    events: list[MoveEvent],
    threshold: float = 0.5,
) -> list[MoveEvent]:
    """Filter move events by confidence threshold.

    Args:
        events: Raw move events from inference.
        threshold: Minimum confidence to keep a move.

    Returns:
        Filtered move events.
    """
    return [e for e in events if e.confidence >= threshold]


def detect_game_completion(
    events: list[MoveEvent],
    no_move_frames: int = 30,
    last_frame: int = 0,
) -> str:
    """Detect if a game has likely ended.

    Heuristics:
    - If the final position is checkmate/stalemate → "completed"
    - If no moves for many frames → "completed" (game likely ended)
    - Otherwise → "in_progress"

    Args:
        events: Move events for the game.
        no_move_frames: Frames without a move to consider game over.
        last_frame: Last frame index processed.

    Returns:
        Status string.
    """
    if not events:
        return "lost_track"

    # Replay to check final position
    sm = GameStateMachine()
    for ev in events:
        if not sm.push_move(ev.move_uci):
            break

    if sm.is_game_over():
        return "completed"

    # Check if there's been a long gap since last move
    last_move_frame = events[-1].frame_idx
    if last_frame - last_move_frame > no_move_frames:
        return "completed"

    return "in_progress"


def validate_and_repair_pgn(
    events: list[MoveEvent],
    initial_fen: str | None = None,
) -> list[MoveEvent]:
    """Validate move sequence and remove any illegal moves.

    With the constrained decoding head, illegal moves should
    never occur. This is a safety net.

    Args:
        events: Move events to validate.
        initial_fen: Starting FEN.

    Returns:
        Validated move events (illegal moves removed).
    """
    sm = GameStateMachine(fen=initial_fen)
    valid: list[MoveEvent] = []

    for ev in events:
        if sm.push_move(ev.move_uci):
            valid.append(MoveEvent(
                board_id=ev.board_id,
                move_uci=ev.move_uci,
                fen_before=ev.fen_before,
                fen_after=sm.get_fen(),
                confidence=ev.confidence,
                frame_idx=ev.frame_idx,
                is_legal=True,
            ))
        # Skip illegal moves silently

    return valid


def postprocess_game_track(
    track: GameTrack,
    confidence_threshold: float = 0.5,
    completion_frames: int = 30,
) -> GameTrack:
    """Full post-processing pipeline for a game track.

    Args:
        track: Raw game track from inference.
        confidence_threshold: Minimum confidence for moves.
        completion_frames: Frames without move to detect completion.

    Returns:
        Post-processed game track.
    """
    # Filter by confidence
    events = confidence_gate_moves(track.moves, confidence_threshold)

    # Validate move legality
    events = validate_and_repair_pgn(events, track.initial_fen)

    # Detect completion
    status = detect_game_completion(events, completion_frames, track.last_frame)

    # Rebuild PGN
    sm = GameStateMachine(fen=track.initial_fen if track.initial_fen else None)
    for ev in events:
        sm.push_move(ev.move_uci)

    return GameTrack(
        board_id=track.board_id,
        moves=events,
        pgn=sm.to_pgn(),
        initial_fen=track.initial_fen,
        final_fen=sm.get_fen(),
        first_frame=track.first_frame,
        last_frame=track.last_frame,
        status=status,
    )
