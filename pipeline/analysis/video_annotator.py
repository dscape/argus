"""Render annotated move overlays onto videos."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.overlay.overlay_move_detector import GameSegment, OverlayDetectedMove

logger = logging.getLogger(__name__)

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
    config: VideoAnalysisConfig,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    scale = config.font_scale
    thick = config.font_thickness

    recent = move_history[-12:] if len(move_history) > 12 else move_history
    move_lines: list[str] = []
    for move in recent:
        ply = move.move_index
        if ply % 2 == 0:
            move_lines.append(f"{ply // 2 + 1}. {move.move_san}")
        elif move_lines:
            move_lines[-1] += f" {move.move_san}"
        else:
            move_lines.append(f"{ply // 2 + 1}... {move.move_san}")

    panel_w = min(int(w * 0.22), 280)
    panel_x = w - panel_w
    overlay = out.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    y = PADDING + 18
    cv2.putText(out, "Argus Analysis", (panel_x + PADDING, y), FONT, scale * 0.55, HEADER_COLOR, thick)
    y += 28

    if current_move is not None:
        ply = current_move.move_index
        move_number = ply // 2 + 1
        dots = "." if ply % 2 == 0 else "..."
        label = f"{move_number}{dots} {current_move.move_san}"
        cv2.putText(out, label, (panel_x + PADDING, y), FONT, scale * 0.7, MOVE_COLOR, thick + 1)
        y += 30

    y += 5
    cv2.line(out, (panel_x + PADDING, y), (w - PADDING, y), (60, 60, 60), 1)
    y += 15

    for line in move_lines:
        cv2.putText(out, line, (panel_x + PADDING, y), FONT, scale * 0.45, TEXT_COLOR, 1)
        y += 20

    return out


def _speak_move(san: str) -> None:
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
    config: VideoAnalysisConfig,
) -> Path:
    """Render an annotated video with detected moves overlaid."""
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

    all_moves: list[OverlayDetectedMove] = []
    for segment in segments:
        all_moves.extend(segment.moves)
    all_moves.sort(key=lambda move: move.timestamp_seconds)

    logger.info(
        "Annotating video: %dx%d @ %.1f fps, %d frames, %d moves",
        width,
        height,
        video_fps,
        total,
        len(all_moves),
    )

    move_history: list[OverlayDetectedMove] = []
    announced: set[int] = set()
    frame_idx = 0
    move_display_secs = 3.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / video_fps
        while len(move_history) < len(all_moves):
            next_move = all_moves[len(move_history)]
            if next_move.timestamp_seconds <= timestamp:
                move_history.append(next_move)
            else:
                break

        current_move = None
        if move_history:
            last_move = move_history[-1]
            if timestamp - last_move.timestamp_seconds < move_display_secs:
                current_move = last_move

        if current_move and config.tts and current_move.move_index not in announced:
            _speak_move(current_move.move_san)
            announced.add(current_move.move_index)

        writer.write(_draw_move_overlay(frame, current_move, move_history, config))
        frame_idx += 1

    cap.release()
    writer.release()

    logger.info("Annotated video saved: %s (%d frames)", output_path, frame_idx)
    return output_path
