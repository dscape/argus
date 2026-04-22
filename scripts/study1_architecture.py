#!/usr/bin/env python3
"""Study 1: chesscog backbones × argus-projection crops.

Evaluates a pair of chesscog-style checkpoints (trained on argus-projected
PNGs) on argus val, using argus's `extract_projected_*_crop` at inference.
Compares to the argus baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from pipeline.physical.chesscog_baseline.argus_geom_reader import (
    read_board_chesscog_on_argus_crops,
)
from pipeline.physical.chesscog_baseline.dataset_export import (
    _resolve_full_frame_and_corners,
)
from pipeline.physical.chesscog_baseline.metrics import (
    EvalMetrics,
    MetricAggregator,
    render_side_by_side,
)
from pipeline.physical.chesscog_baseline.reader import load_chesscog_classifier
from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.two_stage.classifier_data import NativeFrameLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_device(s: str | None) -> torch.device:
    if s:
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation-root", type=Path, default=_PROJECT_ROOT / "data/physical/val")
    ap.add_argument("--occupancy-ckpt", type=Path, required=True)
    ap.add_argument("--occupancy-cfg", type=Path, required=True)
    ap.add_argument("--piece-ckpt", type=Path, required=True)
    ap.add_argument("--piece-cfg", type=Path, required=True)
    ap.add_argument("--occupancy-size", type=int, default=100)
    ap.add_argument("--piece-size", type=int, default=200)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", type=str, default="resnet18_argus_geom")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    out_dir = args.output_dir or (
        _PROJECT_ROOT / "outputs" / dt.date.today().isoformat() / "study1_architecture"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    occ_cls = load_chesscog_classifier(args.occupancy_ckpt, args.occupancy_cfg, device)
    piece_cls = load_chesscog_classifier(args.piece_ckpt, args.piece_cfg, device)

    rows = [
        r
        for r in load_annotated_oblique_rows(args.annotation_root)
        if r.native_corners
        and r.native_image_bbox
        and r.source_video_id
        and r.source_frame_index is not None
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[{args.label}] evaluating on {len(rows)} boards")

    loader = NativeFrameLoader()
    agg = MetricAggregator()
    per_board: list[dict] = []
    t0 = time.time()
    try:
        for idx, row in enumerate(rows):
            try:
                frame_bgr, corners = _resolve_full_frame_and_corners(
                    row, native_loader=loader, clip_cache={}
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {row.annotation_id}: {exc}")
                continue
            result = read_board_chesscog_on_argus_crops(
                frame_bgr,
                corners,
                occupancy=occ_cls,
                piece=piece_cls,
                occupancy_size=args.occupancy_size,
                piece_size=args.piece_size,
            )
            gt = tuple(int(v) for v in row.labels)
            board_correct, board_exact, piece_only = agg.add(result.class_ids, gt)
            per_board.append(
                {
                    "annotation_id": row.annotation_id,
                    "clip_path": row.clip_path,
                    "frame_index": row.frame_index,
                    "squares_correct": board_correct,
                    "board_exact": int(board_exact),
                    "piece_only_exact": int(piece_only),
                }
            )
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{args.label}] {idx + 1}/{len(rows)}: "
                    f"exact={agg.boards_exact}/{agg.board_count} ({elapsed:.0f}s)"
                )
    finally:
        loader.close()

    metrics = agg.finalize()
    (out_dir / f"metrics_{args.label}.json").write_text(json.dumps(asdict(metrics), indent=2))

    baseline_path = _PROJECT_ROOT / "outputs/2026-04-19/chesscog_baseline/argus_metrics.json"
    variants: list = []
    if baseline_path.exists():
        data = json.loads(baseline_path.read_text())
        variants.append(("argus_baseline", EvalMetrics(**data)))
    variants.append((args.label, metrics))
    summary = render_side_by_side(variants, baseline_label="argus_baseline")
    (out_dir / f"summary_{args.label}.txt").write_text(summary)
    print("\n" + summary)

    with (out_dir / f"per_board_{args.label}.tsv").open("w") as f:
        cols = [
            "annotation_id",
            "clip_path",
            "frame_index",
            "squares_correct",
            "board_exact",
            "piece_only_exact",
        ]
        f.write("\t".join(cols) + "\n")
        for row in per_board:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
