"""Video annotation with detected moves.

Renders move notation onto video frames and optionally uses macOS TTS
to announce moves aloud. Produces an annotated .mp4 output file.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from pipeline.mlx.config import MLXPipelineConfig
from pipeline.overlay.overlay_move_detector import GameSegment, OverlayDetectedMove

logger = logging.getLogger(__name__)

# Annotation styling
BG_COLOR = (0, 0, 0)  # Black background for text
TEXT_COLOR = (255, 255, 255)  # White text
MOVE_COLOR = (0, 255, 200)  # Cyan for the current move
FONT = cv2.FONT_HERSHEY_SIMPLEX
PADDING = 10


def _draw_move_overlay(
    frame: np.ndarray,
    current_move: OverlayDetectedMove | None,
    move_history: list[OverlayDetectedMove],
    game_info: str,
    config: MLXPipelineConfig,
) -> np.ndarray:
    """Draw move notation overlay on a frame.

    Args:
        frame: (H, W, 3) BGR frame (OpenCV format).
        current_move: The move being played at this frame, or None.
        move_history: All moves detected so far.
        game_info: Game description string.
        config: Pipeline configuration.

    Returns:
        Frame with annotation overlay.
    """
    h, w = frame.shape[:2]
    out = frame.copy()
    scale = config.font_scale
    thickness = config.font_thickness

    # Build move list text (last 10 moves)
    recent = move_history[-10:] if len(move_history) > 10 else move_history
    move_lines: list[str] = []
    for m in recent:
        ply = m.move_index
        if ply % 2 == 0:
            move_lines.append(f"{ply // 2 + 1}. {m.move_san}")
        else:
            if move_lines:
                move_lines[-1] += f" {m.move_san}"
            else:
                move_lines.append(f"{ply // 2 + 1}... {m.move_san}")

    # Draw semi-transparent background panel on the right
    panel_w = int(w * 0.25)
    panel_x = w - panel_w
    overlay = out.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # Draw game info header
    y_offset = PADDING + 20
    cv2.putText(
        out,
        "MLX Chess Analysis",
        (panel_x + PADDING, y_offset),
        FONT,
        scale * 0.6,
        MOVE_COLOR,
        thickness,
    )
    y_offset += 30

    # Draw current move highlight
    if current_move is not None:
        move_text = f">> {current_move.move_san}"
        cv2.putText(
            out,
            move_text,
            (panel_x + PADDING, y_offset),
            FONT,
            scale * 0.8,
            MOVE_COLOR,
            thickness + 1,
        )
        y_offset += 35

    # Draw move history
    y_offset += 10
    cv2.putText(
        out,
        "Moves:",
        (panel_x + PADDING, y_offset),
        FONT,
        scale * 0.5,
        TEXT_COLOR,
        1,
    )
    y_offset += 25

    for line in move_lines:
        cv2.putText(
            out,
            line,
            (panel_x + PADDING, y_offset),
            FONT,
            scale * 0.5,
            TEXT_COLOR,
            1,
        )
        y_offset += 22

    return out


def _speak_move(san: str) -> None:
    """Announce a move using macOS TTS (non-blocking)."""
    try:
        # Expand common chess notation for natural speech
        text = san
        text = text.replace("O-O-O", "queenside castles")
        text = text.replace("O-O", "kingside castles")
        text = text.replace("+", " check")
        text = text.replace("#", " checkmate")
        text = text.replace("x", " takes ")
        text = text.replace("=", " promotes to ")

        subprocess.Popen(
            ["say", "-r", "200", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # `say` not available (not macOS)


def annotate_video(
    video_path: str | Path,
    segments: list[GameSegment],
    output_path: str | Path,
    config: MLXPipelineConfig,
    game_info: str = "",
) -> Path:
    """Render an annotated video with detected moves overlaid.

    Args:
        video_path: Path to the original video.
        segments: Detected game segments with moves.
        output_path: Path for the output video.
        config: Pipeline configuration.
        game_info: Optional game description for display.

    Returns:
        Path to the annotated video file.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Build a frame_idx -> move lookup for all segments
    all_moves: list[OverlayDetectedMove] = []
    for seg in segments:
        all_moves.extend(seg.moves)
    all_moves.sort(key=lambda m: m.frame_idx)

    # Map frame indices to moves (move is "current" for a window after detection)
    move_at_frame: dict[int, OverlayDetectedMove] = {}
    move_display_frames = int(fps * 3)  # Show move for 3 seconds
    for m in all_moves:
        for f in range(m.frame_idx, m.frame_idx + move_display_frames):
            move_at_frame[f] = m

    logger.info(
        "Annotating video: %dx%d @ %.1f fps, %d frames, %d moves",
        width,
        height,
        fps,
        total,
        len(all_moves),
    )

    move_history: list[OverlayDetectedMove] = []
    announced: set[int] = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track move history
        current_move = move_at_frame.get(frame_idx)
        if current_move and current_move.move_index not in {m.move_index for m in move_history}:
            move_history.append(current_move)

            # TTS announcement
            if config.tts and current_move.move_index not in announced:
                _speak_move(current_move.move_san)
                announced.add(current_move.move_index)

        annotated = _draw_move_overlay(frame, current_move, move_history, game_info, config)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    logger.info("Annotated video saved: %s", output_path)
    return output_path
