#!/usr/bin/env python3
"""Study 3: gap-closing experiments inspired by chesscog paper ideas.

Sub-commands:
  threshold   (3f) sweep occupancy decision threshold, no retrain
  pad-ratio   (3a) sweep occupancy pad_ratio (requires per-ratio retrain)
  shear       (3b) retrain piece classifier with shear augmentation
  unfreeze    (3c) 2-phase DINOv2 fine-tune for piece classifier
  portrait    (3d) retrain piece with 112x224 portrait crops
  mlp-head    (3e) retrain piece with MLP head on DINOv2 features

Each sub-command writes `outputs/<date>/study3_<name>/` with `metrics_*.json`,
`summary.txt`, `per_board.tsv`.

Baseline for deltas is always the current argus weights
(`weights/physical/square_classifier/{occupancy,piece_corrected}/..._classifier.pt`)
evaluated at pad_ratio=0.3 / threshold=0.5 — the configuration used for the
prior chesscog comparison.
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
from pipeline.physical.chesscog_baseline.metrics import (
    EvalMetrics,
    MetricAggregator,
    render_side_by_side,
)
from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.two_stage.classifier_data import NativeFrameLoader
from pipeline.physical.two_stage.reader import read_board

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


def _resolve_full_frame_and_corners(row, *, native_loader: NativeFrameLoader):
    frame = native_loader.load(
        source_video_id=str(row.source_video_id),
        source_frame_index=int(row.source_frame_index),
    )
    x_off, y_off, _, _ = row.native_image_bbox
    corners = tuple((float(c[0] + x_off), float(c[1] + y_off)) for c in row.native_corners)
    return frame, corners


def _evaluate_configuration(
    *,
    annotation_root: Path,
    occupancy_ckpt: Path,
    piece_ckpt: Path,
    device: torch.device,
    occupancy_threshold: float = 0.5,
    occupancy_pad_ratio: float = 0.3,
    occupancy_crop_size: int = 112,
    piece_crop_size: int = 224,
    limit: int = 0,
    label: str = "variant",
    per_board_sink: list[dict] | None = None,
) -> EvalMetrics:
    occ_model = _load_argus_classifier(occupancy_ckpt, device)
    piece_model = _load_argus_classifier(piece_ckpt, device)

    rows = [
        r
        for r in load_annotated_oblique_rows(annotation_root)
        if r.native_corners
        and r.native_image_bbox
        and r.source_video_id
        and r.source_frame_index is not None
    ]
    if limit > 0:
        rows = rows[:limit]
    print(f"[{label}] evaluating on {len(rows)} boards")

    loader = NativeFrameLoader()
    agg = MetricAggregator()
    t0 = time.time()
    try:
        for idx, row in enumerate(rows):
            try:
                frame_bgr, corners = _resolve_full_frame_and_corners(row, native_loader=loader)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {row.annotation_id}: {exc}")
                continue
            result = read_board(
                frame_bgr,
                corners,
                occupancy_model=occ_model,
                piece_model=piece_model,
                occupancy_threshold=occupancy_threshold,
                occupancy_pad_ratio=occupancy_pad_ratio,
                occupancy_crop_size=occupancy_crop_size,
                piece_crop_size=piece_crop_size,
                device=device,
            )
            gt = tuple(int(v) for v in row.labels)
            board_correct, board_exact, piece_only = agg.add(result.class_ids, gt)
            if per_board_sink is not None:
                per_board_sink.append(
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
    finally:
        loader.close()
    return agg.finalize()


def _write_outputs(
    out_dir: Path,
    variants: list[tuple[str, EvalMetrics]],
    per_board_rows_by_label: dict[str, list[dict]],
    *,
    baseline_label: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, metrics in variants:
        safe_label = label.replace("/", "_").replace(" ", "_")
        (out_dir / f"metrics_{safe_label}.json").write_text(json.dumps(asdict(metrics), indent=2))
    summary = render_side_by_side(variants, baseline_label=baseline_label)
    (out_dir / "summary.txt").write_text(summary)
    print("\n" + summary)
    for label, rows in per_board_rows_by_label.items():
        safe_label = label.replace("/", "_").replace(" ", "_")
        path = out_dir / f"per_board_{safe_label}.tsv"
        cols = [
            "annotation_id",
            "clip_path",
            "frame_index",
            "squares_correct",
            "board_exact",
            "piece_only_exact",
        ]
        with path.open("w") as f:
            f.write("\t".join(cols) + "\n")
            for row in rows:
                f.write("\t".join(str(row[c]) for c in cols) + "\n")


# ---------- 3f: threshold sweep ----------


def run_threshold(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    out_dir = args.output_dir or (
        _PROJECT_ROOT / "outputs" / dt.date.today().isoformat() / "study3_threshold"
    )
    thresholds = args.thresholds or [0.5, 0.6, 0.7, 0.8]
    variants: list[tuple[str, EvalMetrics]] = []
    per_board: dict[str, list[dict]] = {}
    for thr in thresholds:
        label = f"thr{thr:.2f}"
        rows: list[dict] = []
        metrics = _evaluate_configuration(
            annotation_root=args.annotation_root,
            occupancy_ckpt=args.occupancy_ckpt,
            piece_ckpt=args.piece_ckpt,
            device=device,
            occupancy_threshold=thr,
            occupancy_pad_ratio=args.pad_ratio,
            limit=args.limit,
            label=label,
            per_board_sink=rows,
        )
        variants.append((label, metrics))
        per_board[label] = rows
    _write_outputs(out_dir, variants, per_board, baseline_label="thr0.50")
    print(f"wrote {out_dir}")


# ---------- argument parsers ----------


def _add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--annotation-root", type=Path, default=_PROJECT_ROOT / "data" / "physical" / "val"
    )
    parser.add_argument(
        "--occupancy-ckpt",
        type=Path,
        default=_PROJECT_ROOT
        / "weights/physical/square_classifier/occupancy/occupancy_classifier.pt",
    )
    parser.add_argument(
        "--piece-ckpt",
        type=Path,
        default=_PROJECT_ROOT
        / "weights/physical/square_classifier/piece_corrected/piece_classifier.pt",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    thr = sub.add_parser("threshold", help="Study 3f: occupancy threshold sweep")
    _add_common_eval_args(thr)
    thr.add_argument("--thresholds", type=float, nargs="+", default=None)
    thr.add_argument("--pad-ratio", type=float, default=0.3)
    thr.set_defaults(func=run_threshold)

    return ap


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
