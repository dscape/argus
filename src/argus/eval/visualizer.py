"""Visualization utilities for debugging and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary


def overlay_predictions_on_frames(
    frames: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    detect_probs: torch.Tensor,
    move_mask: torch.Tensor,
) -> list[np.ndarray]:
    """Create annotated frames showing predictions vs ground truth.

    Args:
        frames: (T, C, H, W) image tensors in [0, 1].
        predictions: (T,) predicted move indices.
        targets: (T,) ground truth move indices.
        detect_probs: (T,) move detection probabilities.
        move_mask: (T,) True at move frames.

    Returns:
        List of annotated (H, W, 3) uint8 numpy arrays.
    """
    try:
        import cv2
    except ImportError:
        return []

    vocab = get_vocabulary()
    annotated: list[np.ndarray] = []

    for t in range(len(frames)):
        # Convert frame to uint8
        frame = frames[t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        H, W = frame.shape[:2]

        # Scale font based on image size
        scale = max(W / 400, 0.3)
        thickness = max(int(scale), 1)

        # Frame number
        cv2.putText(
            frame, f"t={t}", (5, int(15 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, scale * 0.4, (255, 255, 255), thickness
        )

        # Detection probability bar
        bar_w = int(W * 0.3)
        bar_h = int(8 * scale)
        bar_x = W - bar_w - 5
        bar_y = 5
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        fill_w = int(bar_w * detect_probs[t].item())
        color = (0, 255, 0) if detect_probs[t] > 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        # Move info at bottom
        if move_mask[t]:
            gt_idx = targets[t].item()
            gt_uci = vocab.index_to_uci(gt_idx) if gt_idx < vocab.num_moves else "?"
            pred_idx = predictions[t].item()
            pred_uci = vocab.index_to_uci(pred_idx) if pred_idx < vocab.num_moves else "no_move"

            correct = gt_idx == pred_idx
            color = (0, 255, 0) if correct else (0, 0, 255)

            text = f"GT:{gt_uci} Pred:{pred_uci}"
            cv2.putText(
                frame, text, (5, H - int(5 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.35, color, thickness
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated.append(frame)

    return annotated


def save_annotated_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: float = 5.0,
) -> None:
    """Save annotated frames as a video file.

    Args:
        frames: List of (H, W, 3) uint8 numpy arrays.
        output_path: Output video path.
        fps: Frames per second.
    """
    if not frames:
        return

    try:
        import cv2
    except ImportError:
        return

    H, W = frames[0].shape[:2]
    path = str(output_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
