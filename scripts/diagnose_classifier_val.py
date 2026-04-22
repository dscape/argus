#!/usr/bin/env python3
"""Phase B diagnostic: classifier-val vs end-to-end.

For a given piece classifier variant, computes piece-classification accuracy
on four subsets of argus val squares:

  A = all GT-occupied squares (the PieceSquareDataset val metric)
  B = A ∩ (argus occupancy said "occupied") — reader's piece_accuracy_on_correctly_occupied
  C = A \\ B = pieces missed by occupancy (piece classifier didn't run, but
      we can still score what it WOULD have said if we ran it on all GT-occupied squares)
  D = (not-GT-occupied) ∩ (occupancy said occupied) — false-positive occupancy squares
      where the piece classifier then fires

Together these answer: does a higher classifier-val number translate to
end-to-end? If (A - B) is the main gap, then classifier-val vs end-to-end
diverges because the occupancy head pre-filters away hard-to-classify pieces.

Output: JSON with per-subset accuracy and per-class recall on each.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from pipeline.physical.piece_projection import (
    extract_projected_occupancy_crop,
    extract_projected_piece_crop,
)
from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.two_stage.classifier_data import (
    NativeFrameLoader,
    preprocess_square_crop,
)
from pipeline.physical.two_stage.classifiers import (
    SquareClassifier,
    SquareClassifierConfig,
    piece_label_to_square_class,
    square_class_to_piece_label,
)
from pipeline.shared.board_state import SQUARE_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SubsetMetrics:
    name: str
    total: int
    correct: int
    accuracy: float
    per_class_recall: dict[str, float]
    per_class_total: dict[str, int]


def _resolve_device(s: str | None) -> torch.device:
    if s:
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_classifier(checkpoint_path: Path, device: torch.device) -> SquareClassifier:
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


def _resolve_full_frame_and_corners(row, *, native_loader):
    frame = native_loader.load(
        source_video_id=str(row.source_video_id),
        source_frame_index=int(row.source_frame_index),
    )
    x_off, y_off, _, _ = row.native_image_bbox
    corners = tuple((float(c[0] + x_off), float(c[1] + y_off)) for c in row.native_corners)
    return frame, corners


def _compute_subset_metrics(
    name: str,
    predictions: list[
        tuple[int, int]
    ],  # (gt, pred) — both in piece-label space (0..11) or -1 for not applicable
) -> SubsetMetrics:
    filtered = [(g, p) for g, p in predictions if g >= 0 and p >= 0]
    if not filtered:
        return SubsetMetrics(
            name=name, total=0, correct=0, accuracy=0.0, per_class_recall={}, per_class_total={}
        )
    total = len(filtered)
    correct = sum(1 for g, p in filtered if g == p)
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()
    for g, p in filtered:
        gname = SQUARE_CLASS_NAMES[piece_label_to_square_class(g)]
        per_class_total[gname] += 1
        if g == p:
            per_class_correct[gname] += 1
    per_class_recall = {
        gname: (per_class_correct[gname] / per_class_total[gname])
        if per_class_total[gname]
        else 0.0
        for gname in per_class_total
    }
    return SubsetMetrics(
        name=name,
        total=total,
        correct=correct,
        accuracy=correct / total,
        per_class_recall=per_class_recall,
        per_class_total=dict(per_class_total),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--annotation-root",
        type=Path,
        default=_PROJECT_ROOT / "data/physical/val",
    )
    ap.add_argument(
        "--occupancy-ckpt",
        type=Path,
        default=(
            _PROJECT_ROOT / "weights/physical/square_classifier/occupancy/occupancy_classifier.pt"
        ),
    )
    ap.add_argument(
        "--piece-ckpt",
        type=Path,
        required=True,
        help="Path to a piece classifier .pt. Pass piece_corrected to analyze the baseline.",
    )
    ap.add_argument("--occupancy-threshold", type=float, default=0.5)
    ap.add_argument("--occupancy-pad-ratio", type=float, default=0.3)
    ap.add_argument("--occupancy-crop-size", type=int, default=112)
    ap.add_argument("--piece-crop-size", type=int, default=224)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", type=str, default="baseline")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    out_dir = args.output_dir or (
        _PROJECT_ROOT / "outputs" / dt.date.today().isoformat() / "phaseB_classifier_val_diagnostic"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    occ_model = _load_classifier(args.occupancy_ckpt, device)
    piece_model = _load_classifier(args.piece_ckpt, device)

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
    print(f"[{args.label}] diagnosing on {len(rows)} boards")

    loader = NativeFrameLoader()

    # Accumulators for each subset
    # A: all GT-occupied squares (piece label)
    # B: A ∩ (occupancy says occupied)
    # C: A \ B (pieces missed by occupancy)
    # D: (not GT-occupied) ∩ (occupancy says occupied) — "false-positive occupancy" cases
    preds_A: list[tuple[int, int]] = []
    preds_B: list[tuple[int, int]] = []
    preds_C: list[tuple[int, int]] = []
    fp_occ_count = 0  # |D|

    t0 = time.time()
    try:
        for idx, row in enumerate(rows):
            try:
                frame_bgr, corners = _resolve_full_frame_and_corners(row, native_loader=loader)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {row.annotation_id}: {exc}")
                continue
            gt = [int(v) for v in row.labels]

            # 1) Run occupancy classifier on all 64 squares
            occ_crops = [
                extract_projected_occupancy_crop(
                    frame_bgr,
                    corners,
                    row=i // 8,
                    col=i % 8,
                    output_size=args.occupancy_crop_size,
                    pad_ratio=args.occupancy_pad_ratio,
                )
                for i in range(64)
            ]
            occ_tensor = torch.stack(
                [preprocess_square_crop(c, size=args.occupancy_crop_size) for c in occ_crops],
                dim=0,
            ).to(device)
            with torch.no_grad():
                occ_logits = occ_model(occ_tensor)
                occ_probs = torch.softmax(occ_logits, dim=-1)
                occ_said_occupied = (
                    (occ_probs[:, 1] >= args.occupancy_threshold).cpu().numpy().tolist()
                )

            # 2) Run piece classifier on ALL 64 squares (not just occupancy-flagged ones)
            piece_crops = [
                extract_projected_piece_crop(
                    frame_bgr,
                    corners,
                    row=i // 8,
                    col=i % 8,
                    output_size=args.piece_crop_size,
                )
                for i in range(64)
            ]
            piece_tensor = torch.stack(
                [preprocess_square_crop(c, size=args.piece_crop_size) for c in piece_crops],
                dim=0,
            ).to(device)
            with torch.no_grad():
                piece_logits = piece_model(piece_tensor)
                piece_pred = piece_logits.argmax(dim=-1).cpu().numpy().tolist()

            # 3) Aggregate into subsets
            for i in range(64):
                gt_class = gt[i]
                gt_occ = gt_class != 0
                occ_flag = bool(occ_said_occupied[i])

                if gt_occ:
                    gt_piece_label = square_class_to_piece_label(gt_class)
                    p_pred = int(piece_pred[i])
                    preds_A.append((gt_piece_label, p_pred))
                    if occ_flag:
                        preds_B.append((gt_piece_label, p_pred))
                    else:
                        preds_C.append((gt_piece_label, p_pred))
                else:
                    if occ_flag:
                        fp_occ_count += 1

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{args.label}] {idx + 1}/{len(rows)}: "
                    f"|A|={len(preds_A)} |B|={len(preds_B)} |C|={len(preds_C)} |D|={fp_occ_count} "
                    f"({elapsed:.0f}s)"
                )
    finally:
        loader.close()

    metrics_A = _compute_subset_metrics("A (all GT-occupied)", preds_A)
    metrics_B = _compute_subset_metrics("B (A ∩ occ-said-occupied)", preds_B)
    metrics_C = _compute_subset_metrics("C (A \\ B = pieces missed by occupancy)", preds_C)

    # |D| is just a count of false-positive occupancy
    result = {
        "label": args.label,
        "occupancy_threshold": args.occupancy_threshold,
        "occupancy_pad_ratio": args.occupancy_pad_ratio,
        "subsets": {
            "A": {
                "name": metrics_A.name,
                "total": metrics_A.total,
                "correct": metrics_A.correct,
                "accuracy": metrics_A.accuracy,
                "per_class_recall": metrics_A.per_class_recall,
                "per_class_total": metrics_A.per_class_total,
            },
            "B": {
                "name": metrics_B.name,
                "total": metrics_B.total,
                "correct": metrics_B.correct,
                "accuracy": metrics_B.accuracy,
                "per_class_recall": metrics_B.per_class_recall,
                "per_class_total": metrics_B.per_class_total,
            },
            "C": {
                "name": metrics_C.name,
                "total": metrics_C.total,
                "correct": metrics_C.correct,
                "accuracy": metrics_C.accuracy,
                "per_class_recall": metrics_C.per_class_recall,
                "per_class_total": metrics_C.per_class_total,
            },
            "D": {
                "name": "D (false-positive occupancy, no GT piece)",
                "total": fp_occ_count,
                "note": (
                    "count only; piece classifier fires here but there's no GT piece to compare to"
                ),
            },
        },
    }
    (out_dir / f"diagnostic_{args.label}.json").write_text(json.dumps(result, indent=2))

    print(
        f"\n[{args.label}] subset sizes: |A|={metrics_A.total} |B|={metrics_B.total} "
        f"|C|={metrics_C.total} |D|={fp_occ_count}"
    )
    print(
        f"[{args.label}] A accuracy: {metrics_A.accuracy:.4f} "
        f"({metrics_A.correct}/{metrics_A.total})"
    )
    print(
        f"[{args.label}] B accuracy: {metrics_B.accuracy:.4f} "
        f"({metrics_B.correct}/{metrics_B.total})"
    )
    print(
        f"[{args.label}] C accuracy: {metrics_C.accuracy:.4f} "
        f"({metrics_C.correct}/{metrics_C.total})"
    )
    print(f"\nwrote {out_dir / f'diagnostic_{args.label}.json'}")


if __name__ == "__main__":
    main()
