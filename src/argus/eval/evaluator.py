"""End-to-end evaluation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from argus.chess.move_vocabulary import NO_MOVE_IDX, get_vocabulary
from argus.eval.metrics import (
    compute_move_accuracy_topk,
    compute_move_metrics,
    pgn_edit_distance,
    prefix_accuracy,
)
from argus.model.argus_model import ArgusModel

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Aggregated evaluation results."""

    move_accuracy: float = 0.0
    move_accuracy_top5: float = 0.0
    move_detection_precision: float = 0.0
    move_detection_recall: float = 0.0
    move_detection_f1: float = 0.0
    illegal_move_rate: float = 0.0
    avg_pgn_edit_distance: float = 0.0
    avg_prefix_accuracy: float = 0.0
    num_games: int = 0
    per_game_results: list[dict[str, float]] = field(default_factory=list)


class Evaluator:
    """Runs end-to-end evaluation on a dataset.

    Processes clips through the model, extracts predicted move sequences,
    and computes all metrics against ground truth.
    """

    def __init__(
        self,
        model: ArgusModel,
        device: str | torch.device = "cuda",
        frame_tolerance: int = 3,
        detect_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.frame_tolerance = frame_tolerance
        self.detect_threshold = detect_threshold
        self.vocab = get_vocabulary()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> EvalResult:  # type: ignore[type-arg]
        """Run full evaluation.

        Args:
            dataloader: DataLoader yielding batched clips with ground truth.

        Returns:
            EvalResult with aggregated metrics.
        """
        self.model.eval()
        self.model.to(self.device)

        all_move_accs: list[float] = []
        all_move_top5: list[float] = []
        all_detect_prec: list[float] = []
        all_detect_rec: list[float] = []
        all_detect_f1: list[float] = []
        all_ped: list[float] = []
        all_pa: list[float] = []
        per_game: list[dict[str, float]] = []

        for batch in dataloader:
            frames = batch["frames"].to(self.device)
            move_targets = batch["move_targets"].to(self.device)
            detect_targets = batch["detect_targets"].to(self.device)
            legal_masks = batch["legal_masks"].to(self.device)
            move_mask = batch["move_mask"].to(self.device)

            output = self.model(crops=frames, legal_masks=legal_masks)

            move_logits = output.move_logits.squeeze(2)
            move_probs = output.move_probs.squeeze(2)
            detect_logits = output.detect_logits.squeeze(2)

            preds = move_probs.argmax(dim=-1)

            # Per-batch move metrics
            batch_metrics = compute_move_metrics(
                preds, move_targets, detect_logits, detect_targets, move_mask
            )
            all_move_accs.append(batch_metrics["move_accuracy"])
            all_detect_prec.append(batch_metrics["move_detection_precision"])
            all_detect_rec.append(batch_metrics["move_detection_recall"])
            all_detect_f1.append(batch_metrics["move_detection_f1"])

            # Top-5 accuracy
            top5 = compute_move_accuracy_topk(move_logits, move_targets, move_mask, k=5)
            all_move_top5.append(top5)

            # Per-game PGN metrics
            B = preds.shape[0]
            for b in range(B):
                pred_moves = self._extract_move_sequence(preds[b], detect_logits[b])
                gt_moves = self._extract_gt_moves(move_targets[b], move_mask[b])

                ped = pgn_edit_distance(pred_moves, gt_moves)
                pa = prefix_accuracy(pred_moves, gt_moves)
                all_ped.append(ped)
                all_pa.append(pa)

                per_game.append({
                    "pgn_edit_distance": ped,
                    "prefix_accuracy": pa,
                    "num_predicted_moves": len(pred_moves),
                    "num_gt_moves": len(gt_moves),
                })

        def safe_mean(lst: list[float]) -> float:
            return sum(lst) / max(len(lst), 1)

        return EvalResult(
            move_accuracy=safe_mean(all_move_accs),
            move_accuracy_top5=safe_mean(all_move_top5),
            move_detection_precision=safe_mean(all_detect_prec),
            move_detection_recall=safe_mean(all_detect_rec),
            move_detection_f1=safe_mean(all_detect_f1),
            illegal_move_rate=0.0,
            avg_pgn_edit_distance=safe_mean(all_ped),
            avg_prefix_accuracy=safe_mean(all_pa),
            num_games=len(per_game),
            per_game_results=per_game,
        )

    def _extract_move_sequence(
        self,
        predictions: torch.Tensor,
        detect_logits: torch.Tensor,
    ) -> list[str]:
        """Extract predicted move sequence from a single clip.

        Args:
            predictions: (T,) predicted move indices.
            detect_logits: (T,) detection logits.

        Returns:
            List of UCI move strings (excluding NO_MOVE frames).
        """
        moves: list[str] = []
        detect_probs = torch.sigmoid(detect_logits)

        for t in range(len(predictions)):
            if detect_probs[t] > self.detect_threshold:
                idx = predictions[t].item()
                if idx != NO_MOVE_IDX and idx < self.vocab.num_moves:
                    moves.append(self.vocab.index_to_uci(idx))

        return moves

    def _extract_gt_moves(
        self,
        targets: torch.Tensor,
        move_mask: torch.Tensor,
    ) -> list[str]:
        """Extract ground truth move sequence."""
        moves: list[str] = []
        for t in range(len(targets)):
            if move_mask[t]:
                idx = targets[t].item()
                if idx != NO_MOVE_IDX and idx < self.vocab.num_moves:
                    moves.append(self.vocab.index_to_uci(idx))
        return moves
