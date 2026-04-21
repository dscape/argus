#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from study.templates.eval.evaluator import evaluate_template_matching
from study.templates.inference.template_match import load_template_bank


def main() -> None:
    args = build_parser().parse_args()
    template_bank = load_template_bank(args.template_bank)
    template_bank["_loaded_from"] = str(Path(args.template_bank).resolve())
    output_dir = args.output_dir or (_THIS_DIR / args.run_id)
    metrics = evaluate_template_matching(
        template_bank=template_bank,
        proposal_source=args.proposal_source,
        output_dir=output_dir,
        eval_labels=args.eval_labels,
        device=args.device,
        max_frames=args.max_frames,
        source_video_id=args.source_video_id,
        match_threshold=args.match_threshold,
        failures_per_category=args.failures_per_category,
    )
    print(json.dumps(metrics["overall"], indent=2, sort_keys=True))
    print(f"wrote {(Path(output_dir) / 'metrics.json').resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the template-matching study on study/eval."
    )
    parser.add_argument("--template-bank", type=Path, required=True)
    parser.add_argument("--proposal-source", choices=("cuboid", "sam3"), required=True)
    parser.add_argument(
        "--eval-labels",
        type=Path,
        default=_PROJECT_ROOT / "study" / "eval" / "labels.jsonl",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--source-video-id", type=str, default=None)
    parser.add_argument("--match-threshold", type=float, default=0.75)
    parser.add_argument("--failures-per-category", type=int, default=4)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


if __name__ == "__main__":
    main()
