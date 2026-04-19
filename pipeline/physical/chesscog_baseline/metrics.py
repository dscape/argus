"""Shared metric aggregator + summary renderer for argus/chesscog studies.

Originally lived inside `scripts/compare_chesscog_baseline.py`. Extracted so
every study script (3a..3f, study1, study2) can produce identically-shaped
`EvalMetrics` JSON + side-by-side summary.txt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pipeline.shared.board_state import SQUARE_CLASS_NAMES


@dataclass
class EvalMetrics:
    board_count: int
    square_count: int
    per_square_accuracy: float
    empty_accuracy: float
    non_empty_accuracy: float
    occupancy_accuracy: float
    piece_accuracy_on_correctly_occupied: float
    board_exact_match: float
    piece_only_exact_match: float
    mean_squares_correct_per_board: float
    per_class_recall: dict[str, float]
    confusion: dict[str, dict[str, int]]


class MetricAggregator:
    """Streaming accumulator for EvalMetrics across a dataset of 64-square boards."""

    def __init__(self) -> None:
        self.board_count = 0
        self.boards_exact = 0
        self.piece_only_exact = 0
        self.all_correct: list[int] = []
        self.per_square_total = 0
        self.per_square_correct = 0
        self.empty_total = 0
        self.empty_correct = 0
        self.non_empty_total = 0
        self.non_empty_correct = 0
        self.occupancy_correct = 0
        self.piece_on_correct_occupied_total = 0
        self.piece_on_correct_occupied_correct = 0
        self.confusion: dict[str, Counter] = {name: Counter() for name in SQUARE_CLASS_NAMES}
        self.per_class_total: Counter = Counter()

    def add(
        self, predicted: tuple[int, ...], gt: tuple[int, ...]
    ) -> tuple[int, bool, bool]:
        board_correct = sum(int(p == g) for p, g in zip(predicted, gt))
        self.all_correct.append(board_correct)
        self.per_square_total += 64
        self.per_square_correct += board_correct
        board_exact = board_correct == 64
        if board_exact:
            self.boards_exact += 1
        piece_squares = [i for i, g in enumerate(gt) if g != 0]
        piece_only = bool(piece_squares) and all(
            predicted[i] == gt[i] for i in piece_squares
        )
        if piece_only:
            self.piece_only_exact += 1
        for p, g in zip(predicted, gt):
            g_name = SQUARE_CLASS_NAMES[g]
            p_name = SQUARE_CLASS_NAMES[p]
            self.confusion[g_name][p_name] += 1
            self.per_class_total[g_name] += 1
            if g == 0:
                self.empty_total += 1
                if p == 0:
                    self.empty_correct += 1
                    self.occupancy_correct += 1
            else:
                self.non_empty_total += 1
                if p == g:
                    self.non_empty_correct += 1
                if p != 0:
                    self.occupancy_correct += 1
                    self.piece_on_correct_occupied_total += 1
                    if p == g:
                        self.piece_on_correct_occupied_correct += 1
        self.board_count += 1
        return board_correct, board_exact, piece_only

    def finalize(self) -> EvalMetrics:
        def safe(n: int, d: int) -> float:
            return float(n) / float(d) if d else 0.0

        per_class_recall = {
            name: safe(self.confusion[name][name], self.per_class_total[name])
            for name in SQUARE_CLASS_NAMES
        }
        confusion = {name: dict(self.confusion[name]) for name in SQUARE_CLASS_NAMES}
        return EvalMetrics(
            board_count=self.board_count,
            square_count=self.per_square_total,
            per_square_accuracy=safe(self.per_square_correct, self.per_square_total),
            empty_accuracy=safe(self.empty_correct, self.empty_total),
            non_empty_accuracy=safe(self.non_empty_correct, self.non_empty_total),
            occupancy_accuracy=safe(self.occupancy_correct, self.per_square_total),
            piece_accuracy_on_correctly_occupied=safe(
                self.piece_on_correct_occupied_correct,
                self.piece_on_correct_occupied_total,
            ),
            board_exact_match=safe(self.boards_exact, self.board_count),
            piece_only_exact_match=safe(self.piece_only_exact, self.board_count),
            mean_squares_correct_per_board=(
                float(np.mean(self.all_correct)) if self.all_correct else 0.0
            ),
            per_class_recall=per_class_recall,
            confusion=confusion,
        )


_SCALAR_FIELDS = [
    "per_square_accuracy",
    "empty_accuracy",
    "non_empty_accuracy",
    "occupancy_accuracy",
    "piece_accuracy_on_correctly_occupied",
    "board_exact_match",
    "piece_only_exact_match",
    "mean_squares_correct_per_board",
]


def render_side_by_side(
    variants: list[tuple[str, EvalMetrics]],
    *,
    baseline_label: str | None = None,
) -> str:
    """Human-readable table with per-metric deltas vs. the first (or named) variant.

    variants: [(label, metrics), ...] in display order.
    baseline_label: if set, all deltas are vs. this label; else vs. variants[0].
    """
    if not variants:
        return "no variants\n"
    labels = [label for label, _ in variants]
    metrics = {label: m for label, m in variants}
    baseline = baseline_label or labels[0]
    base = metrics[baseline]
    n_cols = len(labels)

    lines: list[str] = []
    lines.append(
        f"Compared on {base.board_count} boards ({base.square_count} squares)"
    )
    lines.append(f"baseline = {baseline}")
    lines.append("")
    header_fmt = "{:40s}" + (" {:>12s}" * n_cols) + (" {:>12s}" * (n_cols - 1))
    header = header_fmt.format(
        "metric", *labels, *[f"Δ {l}" for l in labels if l != baseline]
    )
    lines.append(header)
    lines.append("-" * len(header))
    for field in _SCALAR_FIELDS:
        vals = [getattr(metrics[l], field) for l in labels]
        deltas = [
            getattr(metrics[l], field) - getattr(base, field)
            for l in labels
            if l != baseline
        ]
        lines.append(
            header_fmt.format(
                field,
                *[f"{v:.4f}" for v in vals],
                *[f"{d:+.4f}" for d in deltas],
            )
        )
    lines.append("")
    lines.append("per-class recall:")
    lines.append("-" * len(header))
    for cls in SQUARE_CLASS_NAMES:
        vals = [metrics[l].per_class_recall.get(cls, 0.0) for l in labels]
        deltas = [
            metrics[l].per_class_recall.get(cls, 0.0) - base.per_class_recall.get(cls, 0.0)
            for l in labels
            if l != baseline
        ]
        lines.append(
            header_fmt.format(
                cls,
                *[f"{v:.4f}" for v in vals],
                *[f"{d:+.4f}" for d in deltas],
            )
        )
    return "\n".join(lines) + "\n"


__all__ = ["EvalMetrics", "MetricAggregator", "render_side_by_side"]
