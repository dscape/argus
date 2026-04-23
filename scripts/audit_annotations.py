#!/usr/bin/env python3
"""Audit annotation ROI for the physical-board stack.

Reports, per `source_channel_handle`, how many clips sit in
`data/argus/train_real/` vs. how many annotated frames exist in each of
the train and val splits, and recommends the next action for each channel.

The board probe's corner templates are inferred from the val split
(`pipeline/physical/shared/real_board_data.py`). Channels with no val
annotations are unusable by the probe no matter how many train frames
get labeled — `needs_val` rows surface that hazard first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.shared.annotation_coverage import (
    DEFAULT_CLIPS_DIR,
    DEFAULT_MIN_VAL_COVERAGE,
    DEFAULT_TRAIN_ANNOTATIONS,
    DEFAULT_TRAIN_COVERAGE_TARGET,
    DEFAULT_VAL_ANNOTATIONS,
    ROI_NEEDS_TRAIN,
    ROI_NEEDS_VAL,
    compute_coverage,
)
from pipeline.physical.shared.training_receipts import read_training_receipt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit physical-board annotation ROI")
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--train-annotations", type=Path, default=DEFAULT_TRAIN_ANNOTATIONS)
    parser.add_argument("--val-annotations", type=Path, default=DEFAULT_VAL_ANNOTATIONS)
    parser.add_argument(
        "--train-coverage-target",
        type=int,
        default=DEFAULT_TRAIN_COVERAGE_TARGET,
        help="Train frames per channel above which ROI flips to add_diversity",
    )
    parser.add_argument(
        "--min-val-coverage",
        type=int,
        default=DEFAULT_MIN_VAL_COVERAGE,
        help="Val frames per channel required before probe can use the channel",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        action="append",
        default=[],
        help="Path to a trainer's used_manifest.jsonl; repeat to cross-check multiple trainers",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table"
    )
    args = parser.parse_args()

    report = compute_coverage(
        clips_dir=args.clips_dir,
        train_annotations_path=args.train_annotations,
        val_annotations_path=args.val_annotations,
        train_coverage_target=args.train_coverage_target,
        min_val_coverage=args.min_val_coverage,
    )

    consumed = _load_receipts(args.receipt)
    unused_annotations = _unused_annotations(args.train_annotations, consumed)

    if args.json:
        print(
            json.dumps(
                {
                    "channels": report.to_records(),
                    "clips_without_channel": list(report.clips_without_channel),
                    "train_total": report.train_total,
                    "val_total": report.val_total,
                    "receipts": [str(path) for path in args.receipt],
                    "annotations_not_in_receipts": unused_annotations,
                },
                indent=2,
            )
        )
        return

    _print_table(report, args.train_coverage_target, args.min_val_coverage)
    _print_totals(report, unused_annotations, args.receipt)


def _print_table(report, train_coverage_target: int, min_val_coverage: int) -> None:
    header = ("channel", "clips", "val", "train", "probe", "action")
    widths = [max(len(header[i]), 8) for i in range(len(header))]
    for row in report.channels:
        widths[0] = max(widths[0], len(row.channel))

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for row in report.channels:
        print(
            fmt.format(
                row.channel,
                row.clips,
                row.val_annotated_frames,
                row.train_annotated_frames,
                row.probe_usable_clips,
                row.roi_action,
            )
        )
    print()
    print(
        f"rules: {ROI_NEEDS_VAL} when val < {min_val_coverage} "
        f"(probe template missing); {ROI_NEEDS_TRAIN} when val ok but train < "
        f"{train_coverage_target}; add_diversity otherwise."
    )


def _print_totals(report, unused_annotations: list[dict[str, object]], receipts) -> None:
    print()
    print(f"train annotations on disk: {report.train_total}")
    print(f"val annotations on disk:   {report.val_total}")
    if report.clips_without_channel:
        print(
            f"clips missing source_channel_handle: {len(report.clips_without_channel)}"
            " (skipped by probe)"
        )
    if receipts:
        print(f"checked against {len(receipts)} receipt(s): " + ", ".join(str(r) for r in receipts))
        print(f"annotations not referenced by any receipt: {len(unused_annotations)}")
        for row in unused_annotations[:10]:
            print(f"  - {row['annotation_id']} ({row['clip_path']})")
        if len(unused_annotations) > 10:
            print(f"  ... and {len(unused_annotations) - 10} more")


def _load_receipts(paths: list[Path]) -> dict[str, set[str]]:
    consumed: dict[str, set[str]] = {"clip_path": set(), "annotation_id": set()}
    for path in paths:
        for row in read_training_receipt(path):
            for key in consumed:
                value = row.get(key)
                if isinstance(value, str):
                    consumed[key].add(value)
    return consumed


def _unused_annotations(
    train_annotations_path: Path,
    consumed: dict[str, set[str]],
) -> list[dict[str, object]]:
    if not train_annotations_path.exists():
        return []
    if not consumed["annotation_id"] and not consumed["clip_path"]:
        return []
    unused: list[dict[str, object]] = []
    for line in train_annotations_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        annotation_id = record.get("annotation_id")
        clip_path = record.get("clip_path")
        if annotation_id in consumed["annotation_id"]:
            continue
        if clip_path in consumed["clip_path"]:
            continue
        unused.append({"annotation_id": annotation_id, "clip_path": clip_path})
    return unused


if __name__ == "__main__":
    main()
