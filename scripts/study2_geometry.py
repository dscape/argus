#!/usr/bin/env python3
"""Study 2: argus DINOv2 backbone × chesscog warp+crop geometry.

Evaluation phase only (training is delegated to `train_square_classifier.py
--source chesscog-png`). Runs the chesscog-crop + argus-DINOv2 pipeline on
argus val and compares to the argus baseline.
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
from pipeline.physical.chesscog_baseline.argus_on_chesscog_reader import (
    read_board_argus_on_chesscog_crops,
)
from pipeline.physical.chesscog_baseline.dataset_export import (
    _resolve_full_frame_and_corners,
)
from pipeline.physical.chesscog_baseline.metrics import (
    MetricAggregator,
    render_side_by_side,
)
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


def _load_argus_classifier(checkpoint_path: Path, device: torch.device):
    from pipeline.physical.two_stage.classifiers import (
        SquareClassifier,
        SquareClassifierConfig,
    )

    from argus.model.vision_encoder import VisionEncoder

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    encoder_cfg = payload["encoder_config"]
    classifier_cfg = payload["classifier_config"]
    encoder = VisionEncoder(
        encoder_type=encoder_cfg.get("encoder_type", "dinov2"),
        model_name=encoder_cfg.get("model_name", "facebook/dinov2-base"),
        frozen=encoder_cfg.get("frozen", True),
    )
    model = SquareClassifier(
        vision_encoder=encoder,
        config=SquareClassifierConfig(
            num_classes=classifier_cfg["num_classes"],
            dropout=classifier_cfg.get("dropout", 0.1),
        ),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation-root", type=Path, default=_PROJECT_ROOT / "data/physical/val")
    ap.add_argument("--occupancy-ckpt", type=Path, required=True)
    ap.add_argument("--piece-ckpt", type=Path, required=True)
    ap.add_argument("--occupancy-input-size", type=int, default=100)
    ap.add_argument("--piece-input-size", type=int, default=200)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", type=str, default="study2")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    out_dir = args.output_dir or (
        _PROJECT_ROOT / "outputs" / dt.date.today().isoformat() / "study2_geometry"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    occ_model = _load_argus_classifier(args.occupancy_ckpt, device)
    piece_model = _load_argus_classifier(args.piece_ckpt, device)

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
            result = read_board_argus_on_chesscog_crops(
                frame_bgr,
                corners,
                occupancy_model=occ_model,
                piece_model=piece_model,
                occupancy_input_size=args.occupancy_input_size,
                piece_input_size=args.piece_input_size,
                device=device,
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

    # Load argus baseline metrics for side-by-side comparison
    baseline_path = _PROJECT_ROOT / "outputs/2026-04-19/chesscog_baseline/argus_metrics.json"
    variants: list = []
    if baseline_path.exists():
        from pipeline.physical.chesscog_baseline.metrics import EvalMetrics

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
