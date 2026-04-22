#!/usr/bin/env python3
"""One-time prep and fixed evaluation helpers for Argus autoresearch."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_ROOT = Path(__file__).resolve().parent
CACHE_DIR = AUTORESEARCH_ROOT / "cache"
RUNS_DIR = AUTORESEARCH_ROOT / "runs"
SNAPSHOTS_DIR = AUTORESEARCH_ROOT / "snapshots"
CACHE_PATH = CACHE_DIR / "native_realplusmanual_board_logits.pt"
RESULTS_PATH = AUTORESEARCH_ROOT / "results.tsv"
TRAIN_PATH = AUTORESEARCH_ROOT / "train.py"
BEST_TRAIN_PATH = AUTORESEARCH_ROOT / "best_train.py"
BEST_RESULT_PATH = AUTORESEARCH_ROOT / "best_result.json"
BASELINE_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "2026-04-18"
    / "tracker_eval_weights_best_production_native_pieceboxmapped_off.json"
)
RESULTS_HEADER = (
    "snapshot\tboard_exact\tnon_empty_accuracy\tmacro_f1\tmove_recall"
    "\tfalse_change_rate\tstatus\tdescription\n"
)

sys.path.insert(0, str(PROJECT_ROOT))

from argus.device import resolve_device
from pipeline.overlay.replay import build_replay_board
from pipeline.physical.board_probe.board_data import (
    PhysicalEvalBoardDataset,
    load_annotated_board_frame_bgr,
)
from pipeline.physical.board_probe.decoder import decode_sequence_with_production_decoder
from pipeline.physical.board_probe.runtime import read_board_logits_batch_from_frames
from pipeline.physical.board_probe.square_probe import evaluate_probe
from pipeline.shared import (
    LegalMoveStateTracker,
    LookaheadLegalMoveStateTracker,
    SQUARE_CLASS_NAMES,
    SegmentalLegalSequenceDecoder,
    SequenceTrackerFrameResult,
    board_to_class_ids,
)
from scripts.eval_physical_board_tracker import (
    FramePrediction,
    IdentityProbe,
    compute_tracker_sequence_metrics,
)


@dataclass(frozen=True)
class PreparedSequence:
    clip_path: str
    annotation_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    target_labels: torch.Tensor
    logits: torch.Tensor
    initial_board_fen: str
    initial_side_to_move: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PreparedSequence":
        return cls(
            clip_path=str(payload["clip_path"]),
            annotation_ids=tuple(str(value) for value in payload["annotation_ids"]),
            frame_indices=tuple(int(value) for value in payload["frame_indices"]),
            target_labels=torch.as_tensor(payload["target_labels"], dtype=torch.long),
            logits=torch.as_tensor(payload["logits"], dtype=torch.float32),
            initial_board_fen=str(payload["initial_board_fen"]),
            initial_side_to_move=(
                None
                if payload.get("initial_side_to_move") is None
                else str(payload["initial_side_to_move"])
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "clip_path": self.clip_path,
            "annotation_ids": list(self.annotation_ids),
            "frame_indices": list(self.frame_indices),
            "target_labels": self.target_labels.cpu(),
            "logits": self.logits.cpu(),
            "initial_board_fen": self.initial_board_fen,
            "initial_side_to_move": self.initial_side_to_move,
        }


@dataclass(frozen=True)
class PreparedDataset:
    cache_path: Path
    baseline_report_path: Path
    weights_path: Path
    baseline_report: dict[str, Any]
    sequences: tuple[PreparedSequence, ...]

    @property
    def baseline_board_exact(self) -> float:
        return float(self.baseline_report["metrics"]["board_exact_match"])


@dataclass(frozen=True)
class LoggedResult:
    snapshot: str
    board_exact: float
    non_empty_accuracy: float
    macro_f1: float
    move_recall: float
    false_change_rate: float
    status: str
    description: str


@dataclass(frozen=True)
class RunDecision:
    status: str
    snapshot_path: Path
    report_path: Path
    restored_train: bool
    best_snapshot_path: Path


def ensure_workspace() -> None:
    for directory in (CACHE_DIR, RUNS_DIR, SNAPSHOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER)
    elif not RESULTS_PATH.read_text().startswith(RESULTS_HEADER):
        raise ValueError(f"Unexpected results header in {RESULTS_PATH}")
    if TRAIN_PATH.exists() and not BEST_TRAIN_PATH.exists():
        shutil.copy2(TRAIN_PATH, BEST_TRAIN_PATH)


def load_baseline_report() -> dict[str, Any]:
    return json.loads(BASELINE_REPORT_PATH.read_text())


def prepare_dataset(
    *,
    device: str = "auto",
    force: bool = False,
    batch_size: int = 16,
) -> PreparedDataset:
    ensure_workspace()
    if force or not CACHE_PATH.exists():
        dataset = build_cache(device=device, batch_size=batch_size)
    else:
        dataset = load_prepared_dataset()
    return dataset


def build_cache(*, device: str = "auto", batch_size: int = 16) -> PreparedDataset:
    device_name = resolve_device(device)
    baseline_report = load_baseline_report()
    weights_path = PROJECT_ROOT / str(baseline_report["weights_path"])
    dataset = PhysicalEvalBoardDataset()
    rows = sorted(
        dataset.rows,
        key=lambda row: (
            row.clip_path or row.annotation_id,
            -1 if row.frame_index is None else row.frame_index,
            row.annotation_id,
        ),
    )

    rows_by_clip: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        rows_by_clip[row.clip_path or row.annotation_id].append(row)

    clip_cache: dict[Path, dict[str, Any]] = {}
    sequences: list[PreparedSequence] = []
    with torch.inference_mode():
        for clip_key, clip_rows in sorted(rows_by_clip.items()):
            clip_rows.sort(
                key=lambda row: (
                    -1 if row.frame_index is None else row.frame_index,
                    row.annotation_id,
                )
            )
            board_images = [
                load_annotated_board_frame_bgr(row, clip_cache=clip_cache) for row in clip_rows
            ]
            logits_list = read_board_logits_batch_from_frames(
                board_images,
                corners_list=[row.corners for row in clip_rows],
                device=device_name,
                weights_path=weights_path,
                batch_size=batch_size,
            )
            if logits_list is None:
                raise FileNotFoundError(f"Failed to load weights from {weights_path}")
            initial_board_fen, initial_side_to_move = _initial_board_state_for_row(clip_rows[0])
            sequences.append(
                PreparedSequence(
                    clip_path=clip_key,
                    annotation_ids=tuple(str(row.annotation_id) for row in clip_rows),
                    frame_indices=tuple(int(row.frame_index or 0) for row in clip_rows),
                    target_labels=torch.tensor(
                        [[int(value) for value in row.labels] for row in clip_rows],
                        dtype=torch.long,
                    ),
                    logits=torch.stack([logits.cpu() for logits in logits_list], dim=0),
                    initial_board_fen=initial_board_fen,
                    initial_side_to_move=initial_side_to_move,
                )
            )

    payload = {
        "created_at": utc_now(),
        "device": device_name,
        "baseline_report_path": str(BASELINE_REPORT_PATH.relative_to(PROJECT_ROOT)),
        "weights_path": str(weights_path.relative_to(PROJECT_ROOT)),
        "baseline_report": baseline_report,
        "sequences": [sequence.to_payload() for sequence in sequences],
    }
    torch.save(payload, CACHE_PATH)
    return load_prepared_dataset()


def load_prepared_dataset(cache_path: Path = CACHE_PATH) -> PreparedDataset:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run `.venv/bin/python3 autoresearch/prepare.py` first."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    return PreparedDataset(
        cache_path=cache_path,
        baseline_report_path=PROJECT_ROOT / str(payload["baseline_report_path"]),
        weights_path=PROJECT_ROOT / str(payload["weights_path"]),
        baseline_report=dict(payload["baseline_report"]),
        sequences=tuple(
            PreparedSequence.from_payload(sequence_payload)
            for sequence_payload in payload["sequences"]
        ),
    )


def decode_sequence_with_lookahead(
    sequence: PreparedSequence,
    *,
    lookahead_window: int,
    lookahead_margin: float,
) -> list[SequenceTrackerFrameResult]:
    return LookaheadLegalMoveStateTracker(
        sequence.initial_board_fen,
        initial_side_to_move=sequence.initial_side_to_move,
        lookahead_window=lookahead_window,
        move_score_margin=lookahead_margin,
    ).decode(list(sequence.logits))


def decode_sequence_with_segmental(
    sequence: PreparedSequence,
    **decoder_kwargs: Any,
) -> list[SequenceTrackerFrameResult]:
    return list(
        SegmentalLegalSequenceDecoder(
            sequence.initial_board_fen,
            initial_side_to_move=sequence.initial_side_to_move,
            **decoder_kwargs,
        ).decode(sequence.logits).frames
    )


def evaluate_decoder(
    dataset: PreparedDataset,
    *,
    decode_sequence: Callable[[PreparedSequence], list[SequenceTrackerFrameResult]],
    decoder_name: str,
    decoder_config: dict[str, Any],
) -> dict[str, Any]:
    predictions_by_clip: dict[str, list[FramePrediction]] = defaultdict(list)
    square_predictions: list[int] = []
    square_targets: list[int] = []
    board_annotation_ids: list[str] = []

    for sequence in dataset.sequences:
        results = decode_sequence(sequence)
        if len(results) != len(sequence.annotation_ids):
            raise ValueError(
                f"Decoder returned {len(results)} frames for {sequence.clip_path}, "
                f"expected {len(sequence.annotation_ids)}"
            )
        for index, result in enumerate(results):
            predicted_labels = tuple(board_to_class_ids(build_replay_board(result.fen)))
            target_labels = tuple(int(value) for value in sequence.target_labels[index].tolist())
            predictions_by_clip[sequence.clip_path].append(
                FramePrediction(
                    annotation_id=sequence.annotation_ids[index],
                    frame_index=sequence.frame_indices[index],
                    target_labels=target_labels,
                    predicted_labels=predicted_labels,
                    move_uci=result.move_uci,
                )
            )
            square_predictions.extend(predicted_labels)
            square_targets.extend(target_labels)
            board_annotation_ids.extend([sequence.annotation_ids[index]] * 64)

    logits = torch.zeros((len(square_predictions), len(SQUARE_CLASS_NAMES)), dtype=torch.float32)
    for index, class_id in enumerate(square_predictions):
        logits[index, class_id] = 1.0
    metrics = evaluate_probe(
        IdentityProbe(),
        logits,
        torch.tensor(square_targets, dtype=torch.long),
        device=torch.device("cpu"),
        board_annotation_ids=board_annotation_ids,
    )
    move_recall, false_change_rate, sequence_diagnostics = compute_tracker_sequence_metrics(
        predictions_by_clip,
        tolerance=int(dataset.baseline_report.get("move_match_tolerance", 1)),
    )
    return {
        "created_at": utc_now(),
        "prepared_cache": authrelative(dataset.cache_path),
        "baseline_report": projectrelative(dataset.baseline_report_path),
        "weights_path": projectrelative(dataset.weights_path),
        "sequence_count": len(dataset.sequences),
        "evaluated_boards": len(square_predictions) // 64,
        "decoder": {"name": decoder_name, **decoder_config},
        "metrics": metrics.to_dict(),
        "move_detection_recall": move_recall,
        "static_frame_false_change_rate": false_change_rate,
        "sequence_diagnostics": sequence_diagnostics,
    }


def evaluate_with_baseline_tracker(dataset: PreparedDataset) -> dict[str, Any]:
    report = dataset.baseline_report
    tracker_mode = str(report.get("tracker_mode", "lookahead"))
    if tracker_mode == "production":
        decoder_config = report.get("production_decoder")
        return evaluate_decoder(
            dataset,
            decode_sequence=lambda sequence: decode_sequence_with_production_decoder(
                sequence.logits,
                initial_board_fen=sequence.initial_board_fen,
                initial_side_to_move=sequence.initial_side_to_move,
            ),
            decoder_name="production",
            decoder_config=dict(decoder_config) if isinstance(decoder_config, dict) else {},
        )
    if tracker_mode == "lookahead":
        return evaluate_decoder(
            dataset,
            decode_sequence=lambda sequence: decode_sequence_with_lookahead(
                sequence,
                lookahead_window=int(report["lookahead_window"]),
                lookahead_margin=float(report["lookahead_margin"]),
            ),
            decoder_name="lookahead",
            decoder_config={
                "lookahead_window": int(report["lookahead_window"]),
                "lookahead_margin": float(report["lookahead_margin"]),
            },
        )
    if tracker_mode == "greedy":
        move_accept_threshold = float(report.get("move_accept_threshold", 2.5))
        move_accept_margin = float(report.get("move_accept_margin", 0.75))

        def decode_sequence(sequence: PreparedSequence) -> list[SequenceTrackerFrameResult]:
            tracker = LegalMoveStateTracker(
                sequence.initial_board_fen,
                initial_side_to_move=sequence.initial_side_to_move,
                move_accept_threshold=move_accept_threshold,
                move_accept_margin=move_accept_margin,
            )
            return [tracker.update(logits) for logits in sequence.logits]

        return evaluate_decoder(
            dataset,
            decode_sequence=decode_sequence,
            decoder_name="greedy",
            decoder_config={
                "move_accept_threshold": move_accept_threshold,
                "move_accept_margin": move_accept_margin,
            },
        )
    raise ValueError(f"Unsupported baseline tracker_mode: {tracker_mode}")


def verify_baseline(dataset: PreparedDataset, *, tolerance: float = 1e-9) -> dict[str, Any]:
    expected = dataset.baseline_report
    actual = evaluate_with_baseline_tracker(dataset)
    checks = {
        "board_exact_match": abs(
            float(actual["metrics"]["board_exact_match"])
            - float(expected["metrics"]["board_exact_match"])
        ),
        "non_empty_accuracy": abs(
            float(actual["metrics"]["non_empty_accuracy"])
            - float(expected["metrics"]["non_empty_accuracy"])
        ),
        "macro_f1": abs(
            float(actual["metrics"]["macro_f1"])
            - float(expected["metrics"]["macro_f1"])
        ),
        "move_detection_recall": abs(
            float(actual["move_detection_recall"]) - float(expected["move_detection_recall"])
        ),
        "static_frame_false_change_rate": abs(
            float(actual["static_frame_false_change_rate"])
            - float(expected["static_frame_false_change_rate"])
        ),
    }
    max_delta = max(checks.values())
    if max_delta > tolerance:
        raise RuntimeError(
            "Prepared cache does not reproduce the fixed baseline: "
            f"{json.dumps(checks, sort_keys=True)}"
        )
    return actual


def snapshot_current_train(train_path: Path, experiment_name: str) -> Path:
    ensure_workspace()
    snapshot_name = f"{timestamp_slug()}_{slugify(experiment_name)}.py"
    snapshot_path = SNAPSHOTS_DIR / snapshot_name
    shutil.copy2(train_path, snapshot_path)
    return snapshot_path


def write_report_json(report: dict[str, Any], snapshot_path: Path) -> Path:
    report_path = RUNS_DIR / f"{snapshot_path.stem}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report_path


def record_successful_run(
    *,
    train_path: Path,
    snapshot_path: Path,
    report_path: Path,
    report: dict[str, Any],
    description: str,
) -> RunDecision:
    ensure_workspace()
    current_rank = rank_report(report)
    best_rank = current_best_rank()
    status = "keep" if best_rank is None or current_rank > best_rank else "discard"

    _append_results_row(
        LoggedResult(
            snapshot=authrelative(snapshot_path),
            board_exact=float(report["metrics"]["board_exact_match"]),
            non_empty_accuracy=float(report["metrics"]["non_empty_accuracy"]),
            macro_f1=float(report["metrics"]["macro_f1"]),
            move_recall=float(report["move_detection_recall"]),
            false_change_rate=float(report["static_frame_false_change_rate"]),
            status=status,
            description=sanitize_description(description),
        )
    )

    restored_train = False
    if status == "keep":
        shutil.copy2(snapshot_path, BEST_TRAIN_PATH)
        BEST_RESULT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))
    elif BEST_TRAIN_PATH.exists():
        shutil.copy2(BEST_TRAIN_PATH, train_path)
        restored_train = True

    return RunDecision(
        status=status,
        snapshot_path=snapshot_path,
        report_path=report_path,
        restored_train=restored_train,
        best_snapshot_path=BEST_TRAIN_PATH,
    )


def read_logged_results() -> list[LoggedResult]:
    if not RESULTS_PATH.exists():
        return []
    lines = RESULTS_PATH.read_text().splitlines()
    if not lines:
        return []
    rows: list[LoggedResult] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        (
            snapshot,
            board_exact,
            non_empty_accuracy,
            macro_f1,
            move_recall,
            false_change_rate,
            status,
            description,
        ) = line.split("\t", maxsplit=7)
        rows.append(
            LoggedResult(
                snapshot=snapshot,
                board_exact=float(board_exact),
                non_empty_accuracy=float(non_empty_accuracy),
                macro_f1=float(macro_f1),
                move_recall=float(move_recall),
                false_change_rate=float(false_change_rate),
                status=status,
                description=description,
            )
        )
    return rows


def best_logged_result() -> LoggedResult | None:
    results = read_logged_results()
    if not results:
        return None
    return max(results, key=rank_logged_result)


def current_best_rank() -> tuple[float, float, float, float, float] | None:
    if BEST_RESULT_PATH.exists():
        return rank_report(json.loads(BEST_RESULT_PATH.read_text()))
    best_logged = best_logged_result()
    return None if best_logged is None else rank_logged_result(best_logged)


def rank_report(report: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = report["metrics"]
    return (
        float(metrics["board_exact_match"]),
        -float(report["static_frame_false_change_rate"]),
        float(report["move_detection_recall"]),
        float(metrics["macro_f1"]),
        float(metrics["non_empty_accuracy"]),
    )


def rank_logged_result(result: LoggedResult) -> tuple[float, float, float, float, float]:
    return (
        result.board_exact,
        -result.false_change_rate,
        result.move_recall,
        result.macro_f1,
        result.non_empty_accuracy,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return slug.lower() or "experiment"


def projectrelative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def authrelative(path: Path) -> str:
    return str(path.resolve().relative_to(AUTORESEARCH_ROOT.parent))


def sanitize_description(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").strip()


def _append_results_row(result: LoggedResult) -> None:
    line = (
        f"{result.snapshot}\t{result.board_exact:.6f}\t{result.non_empty_accuracy:.6f}"
        f"\t{result.macro_f1:.6f}\t{result.move_recall:.6f}"
        f"\t{result.false_change_rate:.6f}\t{result.status}\t{result.description}\n"
    )
    with RESULTS_PATH.open("a") as handle:
        handle.write(line)


def _initial_board_state_for_row(row: object) -> tuple[str, str | None]:
    clip_path = getattr(row, "clip_path", None)
    if not isinstance(clip_path, str):
        raise ValueError("Held-out eval row is missing clip_path")
    clip = torch.load(PROJECT_ROOT / clip_path, map_location="cpu", weights_only=False)
    initial_board_fen = clip.get("initial_board_fen") if isinstance(clip, dict) else None
    if not isinstance(initial_board_fen, str):
        raise ValueError(f"Clip is missing initial_board_fen: {clip_path}")
    raw_side_to_move = clip.get("initial_side_to_move") if isinstance(clip, dict) else None
    side_to_move = raw_side_to_move if isinstance(raw_side_to_move, str) else None
    return initial_board_fen, side_to_move


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Argus autoresearch workspace")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    dataset = prepare_dataset(device=args.device, force=args.force, batch_size=args.batch_size)
    verified_report = None if args.skip_verify else verify_baseline(dataset)

    print(f"cache_path: {authrelative(dataset.cache_path)}")
    print(f"weights_path: {projectrelative(dataset.weights_path)}")
    print(f"sequence_count: {len(dataset.sequences)}")
    print(
        "board_count: "
        f"{sum(int(sequence.logits.shape[0]) for sequence in dataset.sequences)}"
    )
    print(
        "baseline_tracker_mode: "
        f"{str(dataset.baseline_report.get('tracker_mode', 'lookahead'))}"
    )
    print(
        "baseline_board_exact: "
        f"{float(dataset.baseline_report['metrics']['board_exact_match']):.6f}"
    )
    if verified_report is not None:
        print(
            "verified_board_exact: "
            f"{float(verified_report['metrics']['board_exact_match']):.6f}"
        )
    print(f"results_tsv: {authrelative(RESULTS_PATH)}")
    print(f"best_train: {authrelative(BEST_TRAIN_PATH)}")


if __name__ == "__main__":
    main()
