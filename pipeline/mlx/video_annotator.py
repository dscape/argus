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
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
MOVE_COLOR = (0, 255, 200)
HEADER_COLOR = (100, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PADDING = 12


def _draw_move_overlay(
    frame: np.ndarray,
    current_move: OverlayDetectedMove | None,
    move_history: list[OverlayDetectedMove],
    config: MLXPipelineConfig,
) -> np.ndarray:
    """Draw move notation overlay on a frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    scale = config.font_scale
    thick = config.font_thickness

    # Build formatted move lines (last 12 plies shown)
    recent = move_history[-12:] if len(move_history) > 12 else move_history
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

    # Semi-transparent panel on the right
    panel_w = min(int(w * 0.22), 280)
    panel_x = w - panel_w
    overlay = out.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    y = PADDING + 18
    cv2.putText(out, "MLX Chess", (panel_x + PADDING, y), FONT, scale * 0.55, HEADER_COLOR, thick)
    y += 28

    # Current move highlight
    if current_move is not None:
        ply = current_move.move_index
        num = ply // 2 + 1
        dots = "." if ply % 2 == 0 else "..."
        label = f"{num}{dots} {current_move.move_san}"
        cv2.putText(out, label, (panel_x + PADDING, y), FONT, scale * 0.7, MOVE_COLOR, thick + 1)
        y += 30

    # Divider
    y += 5
    cv2.line(out, (panel_x + PADDING, y), (w - PADDING, y), (60, 60, 60), 1)
    y += 15

    # Move history
    for line in move_lines:
        cv2.putText(out, line, (panel_x + PADDING, y), FONT, scale * 0.45, TEXT_COLOR, 1)
        y += 20

    return out


def _speak_move(san: str) -> None:
    """Announce a move using macOS TTS (non-blocking)."""
    try:
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
        pass


def annotate_video(
    video_path: str | Path,
    segments: list[GameSegment],
    output_path: str | Path,
    config: MLXPipelineConfig,
    game_info: str = "",
) -> Path:
    """Render an annotated video with detected moves overlaid.

    Maps moves to video timestamps (not extracted frame indices) so the
    annotation aligns with the original video regardless of extraction FPS.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))

    # Collect all moves sorted by timestamp
    all_moves: list[OverlayDetectedMove] = []
    for seg in segments:
        all_moves.extend(seg.moves)
    all_moves.sort(key=lambda m: m.timestamp_seconds)

    # Show each move for 3 seconds after its timestamp
    move_display_secs = 3.0

    logger.info(
        "Annotating video: %dx%d @ %.1f fps, %d frames, %d moves",
        width, height, video_fps, total, len(all_moves),
    )

    move_history: list[OverlayDetectedMove] = []
    announced: set[int] = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / video_fps

        # Add any new moves whose timestamp has passed
        while len(move_history) < len(all_moves):
            nxt = all_moves[len(move_history)]
            if nxt.timestamp_seconds <= t:
                move_history.append(nxt)
            else:
                break

        # Current move = most recent move within display window
        current_move = None
        if move_history:
            last = move_history[-1]
            if t - last.timestamp_seconds < move_display_secs:
                current_move = last

        # TTS
        if current_move and config.tts and current_move.move_index not in announced:
            _speak_move(current_move.move_san)
            announced.add(current_move.move_index)

        annotated = _draw_move_overlay(frame, current_move, move_history, config)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    logger.info("Annotated video saved: %s (%d frames)", output_path, frame_idx)
    return output_path
