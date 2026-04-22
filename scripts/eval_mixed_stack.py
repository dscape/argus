#!/usr/bin/env python3
"""Phase A: Evaluate mixed stack (chesscog ResNet18 occupancy + argus DINOv2 piece).

Reuses chesscog ResNet18 occupancy weights from Study 1 and argus's existing
DINOv2 piece classifier. No retraining. Compares end-to-end to the argus baseline
on argus val.
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
from pipeline.physical.chesscog_baseline.dataset_export import (
    _resolve_full_frame_and_corners,
)
from pipeline.physical.chesscog_baseline.metrics import (
    EvalMetrics,
    MetricAggregator,
    render_side_by_side,
)
from pipeline.physical.chesscog_baseline.mixed_stack_reader import (
    read_board_mixed_stack,
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


def _load_argus_piece_classifier(checkpoint_path: Path, device: torch.device):
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
    ap.add_argument(
        "--annotation-root",
        type=Path,
        default=_PROJECT_ROOT / "data/physical/val",
    )
    ap.add_argument(
        "--chesscog-occupancy-ckpt",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "weights/study1_chesscog_on_argus_geom/occupancy/ResNet_20260419-131634"
            / "ResNet_20260419-131634.pt"
        ),
    )
    ap.add_argument(
        "--chesscog-occupancy-cfg",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "weights/study1_chesscog_on_argus_geom/occupancy/ResNet_20260419-131634"
            / "ResNet_20260419-131634.yaml"
        ),
    )
    ap.add_argument(
        "--argus-piece-ckpt",
        type=Path,
        default=(
            _PROJECT_ROOT / "weights/physical/square_classifier/piece_corrected/piece_classifier.pt"
        ),
    )
    ap.add_argument(
        "--occupancy-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7],
    )
    ap.add_argument("--occupancy-crop-size", type=int, default=100)
    ap.add_argument("--piece-crop-size", type=int, default=224)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    print(f"device: {device}")

    out_dir = args.output_dir or (
        _PROJECT_ROOT / "outputs" / dt.date.today().isoformat() / "phaseA_mixed_stack"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    occupancy_cls = load_chesscog_classifier(
        args.chesscog_occupancy_ckpt, args.chesscog_occupancy_cfg, device
    )
    piece_model = _load_argus_piece_classifier(args.argus_piece_ckpt, device)

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
    print(f"evaluating on {len(rows)} boards")

    loader = NativeFrameLoader()
    variants: list[tuple[str, EvalMetrics]] = []
    per_board_by_variant: dict[str, list[dict]] = {}

    try:
        for thr in args.occupancy_thresholds:
            label = f"mixed_thr{thr:.2f}"
            agg = MetricAggregator()
            per_board: list[dict] = []
            t0 = time.time()
            for idx, row in enumerate(rows):
                try:
                    frame_bgr, corners = _resolve_full_frame_and_corners(
                        row, native_loader=loader, clip_cache={}
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {row.annotation_id}: {exc}")
                    continue
                result = read_board_mixed_stack(
                    frame_bgr,
                    corners,
                    occupancy_chesscog=occupancy_cls,
                    piece_argus=piece_model,
                    occupancy_crop_size=args.occupancy_crop_size,
                    piece_crop_size=args.piece_crop_size,
                    occupancy_threshold=thr,
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
                        f"  [{label}] {idx + 1}/{len(rows)}: "
                        f"exact={agg.boards_exact}/{agg.board_count} "
                        f"({elapsed:.0f}s)"
                    )
            metrics = agg.finalize()
            variants.append((label, metrics))
            per_board_by_variant[label] = per_board
            (out_dir / f"metrics_{label}.json").write_text(json.dumps(asdict(metrics), indent=2))
    finally:
        loader.close()

    # Prepend argus baseline for comparison
    baseline_path = _PROJECT_ROOT / "outputs/2026-04-19/chesscog_baseline/argus_metrics.json"
    baseline_variants: list[tuple[str, EvalMetrics]] = []
    if baseline_path.exists():
        data = json.loads(baseline_path.read_text())
        baseline_variants.append(("argus_baseline", EvalMetrics(**data)))

    all_variants = baseline_variants + variants
    summary = render_side_by_side(all_variants, baseline_label="argus_baseline")
    (out_dir / "summary.txt").write_text(summary)
    print("\n" + summary)

    for label, rows_data in per_board_by_variant.items():
        cols = [
            "annotation_id",
            "clip_path",
            "frame_index",
            "squares_correct",
            "board_exact",
            "piece_only_exact",
        ]
        with (out_dir / f"per_board_{label}.tsv").open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows_data:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
