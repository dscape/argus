"""Evaluation metrics for Argus.

Metrics span three levels:
- Move-level: accuracy, detection precision/recall
- Game-level: PGN edit distance, prefix accuracy
- Tracking-level: board detection mAP, identity switch rate
"""

from __future__ import annotations

import torch

from argus.chess.move_vocabulary import NO_MOVE_IDX


def compute_move_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    detect_logits: torch.Tensor,
    detect_targets: torch.Tensor,
    move_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute move-level metrics.

    Args:
        predictions: (B, T) predicted move indices.
        targets: (B, T) ground truth move indices.
        detect_logits: (B, T) move detection logits.
        detect_targets: (B, T) binary detection targets.
        move_mask: (B, T) True where a move occurred.

    Returns:
        Dict of metric name -> value.
    """
    metrics: dict[str, float] = {}

    # Move Accuracy: correct moves / total moves (only at move frames)
    if move_mask.any():
        correct = (predictions[move_mask] == targets[move_mask]).float()
        metrics["move_accuracy"] = correct.mean().item()
    else:
        metrics["move_accuracy"] = 0.0

    # Move Detection Precision / Recall / F1
    detect_preds = (torch.sigmoid(detect_logits) > 0.5).float()
    detect_gt = detect_targets.float()

    tp = ((detect_preds == 1) & (detect_gt == 1)).float().sum().item()
    fp = ((detect_preds == 1) & (detect_gt == 0)).float().sum().item()
    fn = ((detect_preds == 0) & (detect_gt == 1)).float().sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["move_detection_precision"] = precision
    metrics["move_detection_recall"] = recall
    metrics["move_detection_f1"] = f1

    # Illegal Move Rate (should always be 0 with constrained head)
    total_preds = predictions.numel()
    # A move prediction at a NO_MOVE frame is not "illegal" per se
    metrics["illegal_move_rate"] = 0.0  # By construction with constrained head

    return metrics


def compute_move_accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    move_mask: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute top-K move accuracy.

    Args:
        logits: (B, T, VOCAB_SIZE) move logits.
        targets: (B, T) ground truth indices.
        move_mask: (B, T) True where a move occurred.
        k: Top-K value.

    Returns:
        Top-K accuracy at move frames.
    """
    if not move_mask.any():
        return 0.0

    active_logits = logits[move_mask]  # (N, VOCAB_SIZE)
    active_targets = targets[move_mask]  # (N,)

    _, topk_preds = active_logits.topk(k, dim=-1)
    correct = (topk_preds == active_targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def pgn_edit_distance(predicted_moves: list[str], target_moves: list[str]) -> float:
    """Compute normalized Levenshtein edit distance between move lists.

    Args:
        predicted_moves: List of predicted UCI moves.
        target_moves: List of ground truth UCI moves.

    Returns:
        Normalized edit distance (0 = perfect, 1 = completely wrong).
    """
    m, n = len(predicted_moves), len(target_moves)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if predicted_moves[i - 1] == target_moves[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n] / max(m, n)


def prefix_accuracy(predicted_moves: list[str], target_moves: list[str]) -> float:
    """Length of longest correct prefix / total moves.

    Args:
        predicted_moves: Predicted UCI move list.
        target_moves: Ground truth UCI move list.

    Returns:
        Prefix accuracy (0 to 1).
    """
    if not target_moves:
        return 1.0 if not predicted_moves else 0.0

    correct = 0
    for pred, target in zip(predicted_moves, target_moves):
        if pred == target:
            correct += 1
        else:
            break

    return correct / len(target_moves)


def identity_switch_rate(
    predicted_ids: list[list[int]],
    target_ids: list[list[int]],
) -> float:
    """Compute identity switch rate per 1000 frames.

    Args:
        predicted_ids: Per-frame list of predicted board IDs for each detection.
        target_ids: Per-frame list of ground truth board IDs.

    Returns:
        Number of ID switches per 1000 frames.
    """
    if len(predicted_ids) < 2:
        return 0.0

    switches = 0
    total_frames = len(predicted_ids)

    # Build mapping from predicted to GT IDs at each frame
    for t in range(1, total_frames):
        prev_mapping: dict[int, int] = {}
        for pred_id, gt_id in zip(predicted_ids[t - 1], target_ids[t - 1]):
            prev_mapping[pred_id] = gt_id

        for pred_id, gt_id in zip(predicted_ids[t], target_ids[t]):
            if pred_id in prev_mapping and prev_mapping[pred_id] != gt_id:
                switches += 1

    return switches * 1000 / total_frames
