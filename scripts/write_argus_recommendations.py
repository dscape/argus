#!/usr/bin/env python3
"""Aggregate study results into a ranked recommendation document.

Reads every `metrics_*.json` under `outputs/<date>/study*/` and produces
`outputs/<date>/argus_recommendations.md` with deltas vs. the argus baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.chesscog_baseline.metrics import EvalMetrics

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASELINE_PATH = _PROJECT_ROOT / "outputs/2026-04-19/chesscog_baseline/argus_metrics.json"


def _load(path: Path) -> EvalMetrics:
    data = json.loads(path.read_text())
    return EvalMetrics(**data)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _delta_pct(x: float, base: float) -> str:
    d = (x - base) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}pp"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=_PROJECT_ROOT / "outputs")
    ap.add_argument("--date", type=str, default=dt.date.today().isoformat())
    args = ap.parse_args()

    baseline = _load(_BASELINE_PATH)

    # Collect all (label, metrics) tuples from every study under this date.
    day_root = args.output_root / args.date
    variants: list[tuple[str, EvalMetrics, Path]] = []
    if day_root.exists():
        for study_dir in sorted(day_root.iterdir()):
            if not study_dir.is_dir():
                continue
            for mjson in sorted(study_dir.glob("metrics_*.json")):
                label = f"{study_dir.name}/{mjson.stem.removeprefix('metrics_')}"
                try:
                    m = _load(mjson)
                except Exception as exc:  # noqa: BLE001
                    print(f"skip {mjson}: {exc}")
                    continue
                variants.append((label, m, mjson))

    # Rank by per_square_accuracy delta vs baseline
    ranked = sorted(
        variants,
        key=lambda v: v[1].per_square_accuracy - baseline.per_square_accuracy,
        reverse=True,
    )

    out_path = day_root / "argus_recommendations.md"
    lines: list[str] = []
    lines.append(f"# Argus classifier improvements — findings ({args.date})")
    lines.append("")
    lines.append(
        "Baseline: argus DINOv2 + 3D-box-projection classifier, evaluated on "
        f"{baseline.board_count} val boards. Reference: "
        "[outputs/2026-04-19/chesscog_baseline/argus_metrics.json]("
        "outputs/2026-04-19/chesscog_baseline/argus_metrics.json)"
    )
    lines.append("")
    lines.append("| metric | baseline | " + " | ".join(f"{label}" for label, _, _ in ranked) + " |")
    lines.append("|---|---|" + "---|" * len(ranked))
    metric_fields = [
        ("per_square_accuracy", "per-square acc"),
        ("empty_accuracy", "empty acc"),
        ("non_empty_accuracy", "non-empty acc"),
        ("occupancy_accuracy", "occupancy acc"),
        ("piece_accuracy_on_correctly_occupied", "piece acc@occ"),
        ("board_exact_match", "board exact"),
        ("piece_only_exact_match", "piece-only exact"),
        ("mean_squares_correct_per_board", "mean sq correct"),
    ]
    for field, pretty in metric_fields:
        base_val = getattr(baseline, field)
        row_cells = [pretty, f"{base_val:.4f}"]
        for _, m, _ in ranked:
            v = getattr(m, field)
            delta = v - base_val
            sign = "+" if delta >= 0 else ""
            row_cells.append(f"{v:.4f} ({sign}{delta:.4f})")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    lines.append("## Ranking (by per-square accuracy delta)")
    lines.append("")
    for rank, (label, m, path) in enumerate(ranked, 1):
        delta = m.per_square_accuracy - baseline.per_square_accuracy
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{rank}. **{label}** — per-square {_pct(m.per_square_accuracy)} "
            f"({sign}{delta * 100:.2f}pp). "
            f"[metrics]({path.relative_to(_PROJECT_ROOT)})"
        )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines[:50]))


if __name__ == "__main__":
    main()
