#!/usr/bin/env python3
"""Run a logged physical board-probe experiment sweep.

This is a small autoresearch-style harness around `train_physical_board_probe.py`:
it runs a fixed set of experiments, stores each run in its own output directory,
and maintains `results.tsv` plus a short summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_SCRIPT = _PROJECT_ROOT / "scripts" / "train_physical_board_probe.py"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_board_probe_autoresearch"


@dataclass(frozen=True)
class Experiment:
    name: str
    args: tuple[str, ...]
    description: str


DEFAULT_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        name="dino_topdown_noaug_224",
        args=(
            "--encoder-type",
            "dinov2",
            "--input-size",
            "224",
            "--synthetic-source",
            "topdown",
            "--epochs",
            "25",
            "--synthetic-train-positions",
            "600",
            "--synthetic-val-positions",
            "150",
            "--batch-size",
            "32",
        ),
        description="DINO honest baseline without augmentation.",
    ),
    Experiment(
        name="dino_topdown_aug_224",
        args=(
            "--encoder-type",
            "dinov2",
            "--input-size",
            "224",
            "--synthetic-source",
            "topdown",
            "--augment",
            "--epochs",
            "25",
            "--synthetic-train-positions",
            "600",
            "--synthetic-val-positions",
            "150",
            "--batch-size",
            "32",
        ),
        description="DINO with synthetic blur / rectification artifacts.",
    ),
    Experiment(
        name="dino_topdown_aug_weighted_224",
        args=(
            "--encoder-type",
            "dinov2",
            "--input-size",
            "224",
            "--synthetic-source",
            "topdown",
            "--augment",
            "--class-weighting",
            "max_ratio",
            "--epochs",
            "25",
            "--synthetic-train-positions",
            "600",
            "--synthetic-val-positions",
            "150",
            "--batch-size",
            "32",
        ),
        description="DINO with artifact-heavy synthetic data plus class weighting.",
    ),
    Experiment(
        name="yolo_topdown_noaug_224",
        args=(
            "--encoder-type",
            "yolo",
            "--input-size",
            "224",
            "--synthetic-source",
            "topdown",
            "--epochs",
            "25",
            "--synthetic-train-positions",
            "600",
            "--synthetic-val-positions",
            "150",
            "--batch-size",
            "32",
        ),
        description="YOLO honest baseline without augmentation.",
    ),
    Experiment(
        name="yolo_topdown_aug_weighted_224",
        args=(
            "--encoder-type",
            "yolo",
            "--input-size",
            "224",
            "--synthetic-source",
            "topdown",
            "--augment",
            "--class-weighting",
            "max_ratio",
            "--epochs",
            "25",
            "--synthetic-train-positions",
            "600",
            "--synthetic-val-positions",
            "150",
            "--batch-size",
            "32",
        ),
        description="YOLO with artifact-heavy synthetic data plus class weighting.",
    ),
    Experiment(
        name="dino_topdown_aug_weighted_336",
        args=(
            "--encoder-type",
            "dinov2",
            "--input-size",
            "336",
            "--synthetic-source",
            "topdown",
            "--augment",
            "--class-weighting",
            "max_ratio",
            "--epochs",
            "25",
            "--synthetic-train-positions",
            "400",
            "--synthetic-val-positions",
            "100",
            "--batch-size",
            "16",
        ),
        description="Higher-resolution DINO check after the 224 sweeps.",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a logged physical board-probe sweep")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-samples", type=int, default=0)
    parser.add_argument(
        "--experiments",
        type=str,
        default="",
        help="Comma-separated experiment names (default: all).",
    )
    args = parser.parse_args()

    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = select_experiments(args.experiments)
    rows: list[dict[str, str]] = []

    for experiment in experiments:
        output_dir = output_root / experiment.name
        metrics_path = output_dir / "metrics.json"
        command = build_command(
            device=args.device,
            output_dir=output_dir,
            save_samples=args.save_samples,
            experiment=experiment,
        )

        if args.dry_run:
            logger.info("[dry-run] %s", " ".join(command))
            continue

        if args.skip_existing and metrics_path.exists():
            logger.info("Skipping existing experiment %s", experiment.name)
        else:
            logger.info("Running %s", experiment.name)
            subprocess.run(command, check=True, cwd=_PROJECT_ROOT)

        metrics = load_metrics(metrics_path)
        rows.append(result_row(experiment, output_dir, metrics))
        write_results(output_root / "results.tsv", rows)
        write_summary(output_root / "summary.md", rows)

    if args.dry_run:
        return

    logger.info("Wrote sweep results to %s", output_root)


def resolve_output_root(output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


def select_experiments(raw_names: str) -> list[Experiment]:
    if not raw_names.strip():
        return list(DEFAULT_EXPERIMENTS)
    selected_names = [name.strip() for name in raw_names.split(",") if name.strip()]
    by_name = {experiment.name: experiment for experiment in DEFAULT_EXPERIMENTS}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown experiment names: {', '.join(missing)}")
    return [by_name[name] for name in selected_names]


def build_command(
    *,
    device: str,
    output_dir: Path,
    save_samples: int,
    experiment: Experiment,
) -> list[str]:
    command = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
        *experiment.args,
    ]
    if save_samples > 0:
        command.extend(["--save-samples", str(save_samples)])
    return command


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text())


def result_row(
    experiment: Experiment,
    output_dir: Path,
    metrics: dict[str, Any],
) -> dict[str, str]:
    train_metrics = metrics.get("train_metrics", {})
    synth_metrics = metrics.get("synthetic_val_metrics", {})
    real_metrics = metrics.get("real_eval_metrics", {})
    return {
        "experiment": experiment.name,
        "description": experiment.description,
        "encoder": str(metrics.get("encoder_type", "")),
        "input_size": str(metrics.get("input_size", "")),
        "synthetic_source": str(metrics.get("synthetic_source", "")),
        "augment": str(metrics.get("augment", "")),
        "class_weighting": str(metrics.get("class_weighting", "")),
        "train_acc": format_float(train_metrics.get("accuracy")),
        "synth_val_acc": format_float(synth_metrics.get("accuracy")),
        "real_acc": format_float(real_metrics.get("accuracy")),
        "real_non_empty_acc": format_float(real_metrics.get("non_empty_accuracy")),
        "real_macro_f1": format_float(real_metrics.get("macro_f1")),
        "real_board_exact": format_float(real_metrics.get("board_exact_match")),
        "output_dir": str(output_dir.relative_to(_PROJECT_ROOT)),
    }


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "experiment",
        "description",
        "encoder",
        "input_size",
        "synthetic_source",
        "augment",
        "class_weighting",
        "train_acc",
        "synth_val_acc",
        "real_acc",
        "real_non_empty_acc",
        "real_macro_f1",
        "real_board_exact",
        "output_dir",
    ]
    lines = ["\t".join(fieldnames)]
    lines.extend("\t".join(row[field] for field in fieldnames) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return

    best_by_macro_f1 = max(rows, key=lambda row: float(row["real_macro_f1"] or 0.0))
    best_by_non_empty = max(rows, key=lambda row: float(row["real_non_empty_acc"] or 0.0))
    best_macro_line = (
        f"- best real macro F1: `{best_by_macro_f1['experiment']}`"
        f" -> `{best_by_macro_f1['real_macro_f1']}`"
    )
    best_non_empty_line = (
        f"- best real non-empty accuracy: `{best_by_non_empty['experiment']}`"
        f" -> `{best_by_non_empty['real_non_empty_acc']}`"
    )
    lines = [
        "# Physical board probe autoresearch sweep",
        "",
        f"- run count: `{len(rows)}`",
        best_macro_line,
        best_non_empty_line,
        "",
        "Use `results.tsv` for the full table.",
    ]
    path.write_text("\n".join(lines) + "\n")


def format_float(value: Any) -> str:
    if value is None:
        return "0.0000"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
