"""Side-by-side evaluation of argus two-stage reader and chesscog-on-argus reader.

Iterates argus val (with the same row filter as
`scripts/eval_two_stage_board_reader.py`), runs both readers on each board,
accumulates `EvalMetrics`-shaped results for each, and writes:

  outputs/<date>/chesscog_baseline/argus_metrics.json
  outputs/<date>/chesscog_baseline/chesscog_metrics.json
  outputs/<date>/chesscog_baseline/summary.txt
  outputs/<date>/chesscog_baseline/per_board.tsv
  outputs/<date>/chesscog_baseline/disagreements.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from pipeline.physical.chesscog_baseline import ensure_chesscog_on_path
from pipeline.physical.chesscog_baseline.dataset_export import (
    _resolve_full_frame_and_corners,
)
from pipeline.physical.chesscog_baseline.reader import (
    load_chesscog_classifier,
    read_board_chesscog,
)
from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows
from pipeline.physical.two_stage.classifier_data import NativeFrameLoader
from pipeline.physical.two_stage.classifiers import (
    SquareClassifier,
    SquareClassifierConfig,
)
from pipeline.physical.two_stage.reader import (
    read_board,
)
from pipeline.shared.board_state import SQUARE_CLASS_NAMES

ensure_chesscog_on_path()


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


class _MetricAggregator:
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

    def add(self, predicted: tuple[int, ...], gt: tuple[int, ...]) -> tuple[int, bool, bool]:
        board_correct = sum(int(p == g) for p, g in zip(predicted, gt))
        self.all_correct.append(board_correct)
        self.per_square_total += 64
        self.per_square_correct += board_correct
        board_exact = board_correct == 64
        if board_exact:
            self.boards_exact += 1
        piece_squares = [i for i, g in enumerate(gt) if g != 0]
        piece_only = bool(piece_squares) and all(predicted[i] == gt[i] for i in piece_squares)
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
        def safe(n, d):
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


def _load_argus_classifier(checkpoint_path: Path, device: torch.device) -> SquareClassifier:
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


def _render_summary(argus: EvalMetrics, chesscog: EvalMetrics) -> str:
    lines: list[str] = []
    lines.append(f"Compared on {argus.board_count} boards ({argus.square_count} squares)")
    lines.append("")
    col = ("argus", "chesscog", "delta (argus - chesscog)")
    fmt = "{:40s} {:>12s} {:>12s} {:>12s}"
    lines.append(fmt.format("metric", *col))
    lines.append("-" * 80)
    scalar_fields = [
        ("per_square_accuracy", argus.per_square_accuracy, chesscog.per_square_accuracy),
        ("empty_accuracy", argus.empty_accuracy, chesscog.empty_accuracy),
        ("non_empty_accuracy", argus.non_empty_accuracy, chesscog.non_empty_accuracy),
        ("occupancy_accuracy", argus.occupancy_accuracy, chesscog.occupancy_accuracy),
        (
            "piece_accuracy_on_correctly_occupied",
            argus.piece_accuracy_on_correctly_occupied,
            chesscog.piece_accuracy_on_correctly_occupied,
        ),
        ("board_exact_match", argus.board_exact_match, chesscog.board_exact_match),
        ("piece_only_exact_match", argus.piece_only_exact_match, chesscog.piece_only_exact_match),
        (
            "mean_squares_correct_per_board",
            argus.mean_squares_correct_per_board,
            chesscog.mean_squares_correct_per_board,
        ),
    ]
    for name, a, c in scalar_fields:
        lines.append(fmt.format(name, f"{a:.4f}", f"{c:.4f}", f"{a - c:+.4f}"))
    lines.append("")
    lines.append("per-class recall (argus | chesscog | delta)")
    lines.append("-" * 80)
    for cls in SQUARE_CLASS_NAMES:
        a = argus.per_class_recall.get(cls, 0.0)
        c = chesscog.per_class_recall.get(cls, 0.0)
        lines.append(fmt.format(cls, f"{a:.4f}", f"{c:.4f}", f"{a - c:+.4f}"))
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotation-root", type=Path, default=Path("data/physical/val"))
    ap.add_argument("--argus-occupancy-ckpt", type=Path, required=True)
    ap.add_argument("--argus-piece-ckpt", type=Path, required=True)
    ap.add_argument("--chesscog-occupancy-ckpt", type=Path, required=True)
    ap.add_argument("--chesscog-piece-ckpt", type=Path, required=True)
    ap.add_argument("--chesscog-occupancy-cfg", type=Path, required=True)
    ap.add_argument("--chesscog-piece-cfg", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frame-cache-capacity", type=int, default=32)
    return ap


def _resolve_device(s: str | None) -> torch.device:
    if s:
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = _build_parser().parse_args()
    device = _resolve_device(args.device)
    print(f"device: {device}")

    output_dir = args.output_dir
    if output_dir is None:
        today = dt.date.today().isoformat()
        output_dir = Path("outputs") / today / "chesscog_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    argus_occ = _load_argus_classifier(args.argus_occupancy_ckpt, device)
    argus_piece = _load_argus_classifier(args.argus_piece_ckpt, device)
    chesscog_occ = load_chesscog_classifier(
        args.chesscog_occupancy_ckpt, args.chesscog_occupancy_cfg, device
    )
    chesscog_piece = load_chesscog_classifier(
        args.chesscog_piece_ckpt, args.chesscog_piece_cfg, device
    )

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
    print(f"evaluating on {len(rows)} boards from {args.annotation_root}")

    loader = NativeFrameLoader(capacity=args.frame_cache_capacity)

    argus_agg = _MetricAggregator()
    chesscog_agg = _MetricAggregator()
    disagreements: list[dict] = []
    per_board_rows: list[dict] = []

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

            argus_result = read_board(
                frame_bgr,
                corners,
                occupancy_model=argus_occ,
                piece_model=argus_piece,
                device=device,
            )
            chesscog_result = read_board_chesscog(
                frame_bgr,
                corners,
                occupancy=chesscog_occ,
                piece=chesscog_piece,
            )

            gt = tuple(int(v) for v in row.labels)
            argus_correct, argus_exact, argus_piece_only = argus_agg.add(argus_result.class_ids, gt)
            cc_correct, cc_exact, cc_piece_only = chesscog_agg.add(chesscog_result.class_ids, gt)

            per_board_rows.append(
                {
                    "annotation_id": row.annotation_id,
                    "clip_path": row.clip_path,
                    "frame_index": row.frame_index,
                    "argus_squares_correct": argus_correct,
                    "chesscog_squares_correct": cc_correct,
                    "argus_board_exact": int(argus_exact),
                    "chesscog_board_exact": int(cc_exact),
                    "argus_piece_only_exact": int(argus_piece_only),
                    "chesscog_piece_only_exact": int(cc_piece_only),
                }
            )

            if argus_result.class_ids != chesscog_result.class_ids:
                diffs = [
                    {
                        "square_index": i,
                        "gt": SQUARE_CLASS_NAMES[gt[i]],
                        "argus": SQUARE_CLASS_NAMES[argus_result.class_ids[i]],
                        "chesscog": SQUARE_CLASS_NAMES[chesscog_result.class_ids[i]],
                    }
                    for i in range(64)
                    if argus_result.class_ids[i] != chesscog_result.class_ids[i]
                ]
                disagreements.append(
                    {
                        "annotation_id": row.annotation_id,
                        "clip_path": row.clip_path,
                        "frame_index": row.frame_index,
                        "num_differences": len(diffs),
                        "differences": diffs[:20],
                    }
                )

            if (idx + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(
                    f"  {idx + 1}/{len(rows)} boards: "
                    f"argus exact={argus_agg.boards_exact}/{argus_agg.board_count} "
                    f"chesscog exact={chesscog_agg.boards_exact}/{chesscog_agg.board_count} "
                    f"({elapsed:.0f}s)"
                )
    finally:
        loader.close()

    argus_metrics = argus_agg.finalize()
    chesscog_metrics = chesscog_agg.finalize()

    (output_dir / "argus_metrics.json").write_text(
        json.dumps(dataclasses.asdict(argus_metrics), indent=2)
    )
    (output_dir / "chesscog_metrics.json").write_text(
        json.dumps(dataclasses.asdict(chesscog_metrics), indent=2)
    )

    summary = _render_summary(argus_metrics, chesscog_metrics)
    (output_dir / "summary.txt").write_text(summary)
    print("\n" + summary)

    with (output_dir / "per_board.tsv").open("w") as f:
        cols = [
            "annotation_id",
            "clip_path",
            "frame_index",
            "argus_squares_correct",
            "chesscog_squares_correct",
            "argus_board_exact",
            "chesscog_board_exact",
            "argus_piece_only_exact",
            "chesscog_piece_only_exact",
        ]
        f.write("\t".join(cols) + "\n")
        for row in per_board_rows:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")

    (output_dir / "disagreements.json").write_text(json.dumps(disagreements, indent=2))

    print(f"\nwrote results to {output_dir}")


if __name__ == "__main__":
    main()
