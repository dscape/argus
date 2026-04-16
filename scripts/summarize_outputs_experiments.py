#!/usr/bin/env python3
"""Summarize top-level outputs artifacts into outputs/experiments.csv."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CSV_PATH = OUTPUTS_ROOT / "experiments.csv"
_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IGNORED_ROOT_ITEMS = {"overlay_classifier", "plan.md"}

COLUMNS = [
    "date",
    "name",
    "relative_path",
    "item_type",
    "kind",
    "source_file",
    "best_variant",
    "variant_count",
    "file_count",
    "summary_excerpt",
    "model_name",
    "architecture",
    "encoder_type",
    "head_type",
    "input_size",
    "trained_at",
    "checkpoint",
    "weights_path",
    "real_train_positions",
    "synthetic_train_positions",
    "eval_positions",
    "square_accuracy",
    "non_empty_accuracy",
    "board_exact_match",
    "macro_f1",
    "move_detection_recall",
    "false_change_rate",
    "selection_metric",
    "selection_score",
    "extra",
]


def main() -> None:
    rows = [summarize_item(date, path) for date, path in iter_experiment_items(OUTPUTS_ROOT)]
    rows.sort(key=lambda row: (row["date"], row["relative_path"]))

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})

    print(f"Wrote {len(rows)} rows to {CSV_PATH}")


def iter_experiment_items(root: Path):
    for child in sorted(root.iterdir()):
        if (
            child.name.startswith(".")
            or child.name == CSV_PATH.name
            or child.name in _IGNORED_ROOT_ITEMS
        ):
            continue
        if child.is_dir() and _DATE_DIR_RE.match(child.name):
            for item in sorted(child.iterdir()):
                if item.name.startswith("."):
                    continue
                yield child.name, item
        else:
            yield "", child


def summarize_item(date: str, path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "date": date,
        "name": path.name,
        "relative_path": relpath(path),
        "item_type": "dir" if path.is_dir() else "file",
        "kind": infer_kind(path.name),
        "source_file": "",
        "best_variant": "",
        "variant_count": "",
        "file_count": 0,
        "summary_excerpt": "",
        "model_name": "",
        "architecture": "",
        "encoder_type": "",
        "head_type": "",
        "input_size": "",
        "trained_at": "",
        "checkpoint": "",
        "weights_path": "",
        "real_train_positions": "",
        "synthetic_train_positions": "",
        "eval_positions": "",
        "square_accuracy": "",
        "non_empty_accuracy": "",
        "board_exact_match": "",
        "macro_f1": "",
        "move_detection_recall": "",
        "false_change_rate": "",
        "selection_metric": "",
        "selection_score": "",
        "extra": "",
    }

    if path.is_file():
        row.update(parse_artifact_file(path))
        return row

    files = [file for file in path.rglob("*") if file.is_file() and not file.name.startswith(".")]
    row["file_count"] = len(files)
    row["summary_excerpt"] = read_summary_excerpt(path / "summary.md")

    primary = choose_primary_artifact(path)
    if primary is not None:
        row["source_file"] = relpath(primary)
        row.update(parse_artifact_file(primary))
    elif files:
        row["source_file"] = relpath(files[0])

    return row


def choose_primary_artifact(path: Path) -> Path | None:
    preferred = [
        path / "metrics.json",
        path / "report.json",
        path / "summary.json",
        path / "results.tsv",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    json_candidates = sorted(
        file
        for file in path.glob("*.json")
        if file.name not in {"manifest.json", "hydra.yaml", "config.yaml", "overrides.yaml"}
    )
    if json_candidates:
        return json_candidates[0]

    return None


def parse_artifact_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return parse_json_artifact(path)
    if path.suffix == ".tsv":
        return parse_tsv_artifact(path)
    if path.suffix == ".log":
        return {"extra": f"log:{path.name}"}
    return {}


def parse_json_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {"extra": f"unparsed_json:{path.name}"}

    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        best_variant, best_payload = select_best_variant_from_list(payload)
        result = extract_metrics(best_payload)
        result["best_variant"] = best_variant
        result["variant_count"] = len(payload)
        return result

    if (
        isinstance(payload, dict)
        and payload
        and all(isinstance(value, dict) for value in payload.values())
    ):
        best_variant, best_payload = select_best_variant_from_dict(payload)
        result = extract_metrics(best_payload)
        result["best_variant"] = best_variant
        result["variant_count"] = len(payload)
        return result

    if isinstance(payload, dict):
        return extract_metrics(payload)

    return {"extra": f"unsupported_json:{path.name}"}


def parse_tsv_artifact(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({key: value for key, value in row.items() if key is not None})

    if not rows:
        return {}

    best_row = max(rows, key=score_variant)
    extras = {
        key: coerce_scalar(value)
        for key, value in best_row.items()
        if key
        not in {
            "board_exact",
            "board_exact_match",
            "square",
            "accuracy",
            "non_empty",
            "non_empty_accuracy",
            "macro",
            "macro_f1",
            "move_recall",
            "move_detection_recall",
            "false_change",
            "static_frame_false_change_rate",
        }
    }
    return {
        "best_variant": best_row.get("name")
        or best_row.get("decoder")
        or best_row.get("margin")
        or "best",
        "variant_count": len(rows),
        "square_accuracy": first_non_null(best_row.get("square"), best_row.get("accuracy")),
        "non_empty_accuracy": first_non_null(
            best_row.get("non_empty"), best_row.get("non_empty_accuracy")
        ),
        "board_exact_match": first_non_null(
            best_row.get("board_exact"), best_row.get("board_exact_match")
        ),
        "macro_f1": first_non_null(best_row.get("macro"), best_row.get("macro_f1")),
        "move_detection_recall": first_non_null(
            best_row.get("move_recall"), best_row.get("move_detection_recall")
        ),
        "false_change_rate": first_non_null(
            best_row.get("false_change"), best_row.get("static_frame_false_change_rate")
        ),
        "extra": compact_json(extras),
    }


def select_best_variant_from_list(items: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    best_item = max(items, key=score_variant)
    variant_name = str(
        best_item.get("name")
        or best_item.get("decoder")
        or best_item.get("state_aware_proposal_passes")
        or "best"
    )
    return variant_name, best_item


def select_best_variant_from_dict(items: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    variant_name, best_item = max(items.items(), key=lambda item: score_variant(item[1]))
    return variant_name, best_item


def extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    probe_config = (
        payload.get("probe_config") if isinstance(payload.get("probe_config"), dict) else {}
    )

    result = {
        "model_name": first_non_null(
            payload.get("model_name"), nested_get(payload, "metadata", "model_name")
        ),
        "architecture": first_non_null(
            payload.get("architecture"), nested_get(payload, "metadata", "architecture")
        ),
        "encoder_type": first_non_null(
            payload.get("encoder_type"), nested_get(payload, "metadata", "encoder_type")
        ),
        "head_type": first_non_null(payload.get("head_type"), probe_config.get("head_type")),
        "input_size": first_non_null(
            payload.get("input_size"), nested_get(payload, "metadata", "input_size")
        ),
        "trained_at": first_non_null(payload.get("trained_at"), payload.get("evaluated_at")),
        "checkpoint": payload.get("checkpoint", ""),
        "weights_path": payload.get("weights_path", ""),
        "real_train_positions": first_non_null(
            payload.get("real_train_positions"), payload.get("real_train_frames")
        ),
        "synthetic_train_positions": first_non_null(
            payload.get("synthetic_train_positions"), payload.get("synthetic_train_frames")
        ),
        "eval_positions": first_non_null(
            payload.get("real_eval_positions"),
            payload.get("eval_frames"),
            payload.get("evaluated_boards"),
            payload.get("sample_size"),
        ),
        "square_accuracy": first_non_null(
            nested_get(payload, "real_eval_metrics", "accuracy"),
            nested_get(payload, "eval_metrics", "accuracy"),
            nested_get(payload, "metrics", "accuracy"),
            payload.get("accuracy"),
            payload.get("square_accuracy"),
        ),
        "non_empty_accuracy": first_non_null(
            nested_get(payload, "real_eval_metrics", "non_empty_accuracy"),
            nested_get(payload, "eval_metrics", "non_empty_accuracy"),
            nested_get(payload, "metrics", "non_empty_accuracy"),
            payload.get("non_empty_accuracy"),
            payload.get("non_empty"),
        ),
        "board_exact_match": first_non_null(
            nested_get(payload, "real_eval_metrics", "board_exact_match"),
            nested_get(payload, "eval_metrics", "board_exact_match"),
            nested_get(payload, "metrics", "board_exact_match"),
            payload.get("board_exact_match"),
            payload.get("board_exact"),
        ),
        "macro_f1": first_non_null(
            nested_get(payload, "real_eval_metrics", "macro_f1"),
            nested_get(payload, "eval_metrics", "macro_f1"),
            nested_get(payload, "metrics", "macro_f1"),
            payload.get("macro_f1"),
            payload.get("macro"),
        ),
        "move_detection_recall": first_non_null(
            payload.get("move_detection_recall"),
            payload.get("move_recall"),
            nested_get(payload, "metrics", "move_detection_recall"),
        ),
        "false_change_rate": first_non_null(
            payload.get("static_frame_false_change_rate"),
            payload.get("false_change"),
            nested_get(payload, "metrics", "static_frame_false_change_rate"),
        ),
        "selection_metric": payload.get("selection_metric", ""),
        "selection_score": payload.get("selection_score", ""),
        "extra": compact_json(interesting_extras(payload)),
    }

    return result


def interesting_extras(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "augment",
        "class_weighting",
        "synthetic_source",
        "real_loss_weight",
        "dropout",
        "lookahead_window",
        "lookahead_margin",
        "move_accept_threshold",
        "move_accept_margin",
        "temporal_mode",
        "tracker_mode",
        "observation_input",
        "passes",
        "drop_worst_frames",
        "state_aware_proposal_passes",
        "detect_th",
        "move_conf",
        "margin",
        "decoder",
        "feature_layer_indices",
        "selection_source_video_ids",
        "train_source_video_ids",
    ]
    extras = {key: payload[key] for key in keys if key in payload}
    return extras


def read_summary_excerpt(path: Path) -> str:
    if not path.exists():
        return ""
    lines = [line.strip() for line in path.read_text().splitlines()]
    bullets = [line.lstrip("- ") for line in lines if line.startswith("-")][:3]
    if bullets:
        return " | ".join(bullets)
    text_lines = [line for line in lines if line and not line.startswith("#")][:2]
    return " | ".join(text_lines)


def score_variant(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    board_exact = to_float(
        first_non_null(payload.get("board_exact_match"), payload.get("board_exact"))
    )
    square_accuracy = to_float(
        first_non_null(
            payload.get("square_accuracy"), payload.get("accuracy"), payload.get("square")
        )
    )
    non_empty_accuracy = to_float(
        first_non_null(payload.get("non_empty_accuracy"), payload.get("non_empty"))
    )
    macro_f1 = to_float(first_non_null(payload.get("macro_f1"), payload.get("macro")))
    return board_exact, square_accuracy, non_empty_accuracy, macro_f1


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def compact_json(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def coerce_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return ""
        try:
            if "." in text or "e" in text.lower():
                return float(text)
            return int(text)
        except ValueError:
            return text
    return str(value)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def infer_kind(name: str) -> str:
    lowered = name.lower()
    if "board_probe" in lowered:
        return "board_probe"
    if "joint_board_reader" in lowered:
        return "joint_board_reader"
    if "move_model" in lowered:
        return "move_model"
    if "runtime" in lowered or "tracker" in lowered:
        return "runtime_eval"
    if "segmental" in lowered or "legal_" in lowered or "native_oblique" in lowered:
        return "decoder_eval"
    if "dataset" in lowered or "export" in lowered:
        return "dataset"
    if "eval" in lowered:
        return "evaluation"
    return "artifact"


if __name__ == "__main__":
    main()
