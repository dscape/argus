"""Service layer for browsing physical board failure-study bundles."""

from __future__ import annotations

import base64
import csv
import json
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from pipeline.paths import PROJECT_ROOT
from pipeline.physical import board_tracker_failure_study as failure_study

_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
_CANONICAL_BUCKETS = [
    "rectification / localization",
    "piece classifier / square evidence",
    "temporal in-between / move execution ambiguity",
    "decoder / wrong legal hypothesis / error propagation",
    "eval or label issue",
    "other / unclear",
]
_MAX_CONTEXT_FRAMES = 30
_DEFAULT_CONTEXT_FRAMES = 10
_DEFAULT_IMAGE_MAX_SIDE = 720


def list_failure_studies() -> list[dict[str, Any]]:
    studies: list[dict[str, Any]] = []
    for summary_path in _candidate_summary_paths():
        try:
            bundle = _load_bundle(summary_path.parent)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, csv.Error):
            continue

        studies.append(
            {
                "path": _relative_to_project(bundle["study_dir"]),
                "label": bundle["study_dir"].name,
                "modified_at": _isoformat_mtime(bundle["summary_path"]),
                "selected_failures": int(bundle["summary"].get("selected_failures", 0)),
                "total_failures": int(bundle["summary"].get("total_failures", 0)),
                "observation_input": str(
                    bundle["summary"].get("config", {}).get("observation_input", "")
                ),
                "tracker_mode": str(
                    bundle["summary"].get("config", {}).get("tracker_mode", "")
                ),
                "report_path": bundle["summary"].get("report_path"),
            }
        )

    studies.sort(
        key=lambda study: (
            0.0
            if study["modified_at"] is None
            else time.mktime(time.strptime(study["modified_at"], "%Y-%m-%dT%H:%M:%S")),
            study["label"],
        ),
        reverse=True,
    )
    return studies


def get_failure_study(study_path: str) -> dict[str, Any]:
    bundle = _load_bundle(_resolve_study_dir(study_path))
    entries = _merged_entries(bundle)
    report_metrics = _load_report_metrics(bundle.get("report_path"))

    return {
        "path": _relative_to_project(bundle["study_dir"]),
        "label": bundle["study_dir"].name,
        "modified_at": _isoformat_mtime(bundle["summary_path"]),
        "summary": bundle["summary"],
        "report_metrics": report_metrics,
        "entries": entries,
        "bucket_options": list(_CANONICAL_BUCKETS),
        "bucket_counts": _bucket_counts(entries),
    }


def get_failure_study_context(
    study_path: str,
    *,
    selected_index: int,
    context_frames: int = _DEFAULT_CONTEXT_FRAMES,
    image_max_side: int = _DEFAULT_IMAGE_MAX_SIDE,
) -> dict[str, Any]:
    if selected_index <= 0:
        raise ValueError(f"selected_index must be > 0, got {selected_index}")
    if context_frames < 0 or context_frames > _MAX_CONTEXT_FRAMES:
        raise ValueError(
            f"context_frames must be between 0 and {_MAX_CONTEXT_FRAMES}, got {context_frames}"
        )
    if image_max_side <= 0:
        raise ValueError(f"image_max_side must be > 0, got {image_max_side}")

    bundle = _load_bundle(_resolve_study_dir(study_path))
    entries = _merged_entries(bundle)
    entry = next(
        (item for item in entries if int(item.get("selected_index", 0)) == selected_index),
        None,
    )
    if entry is None:
        raise FileNotFoundError(
            f"Failure-study entry {selected_index} not found in {study_path}"
        )

    config = bundle["summary"].get("config", {})
    observation_input = cast(
        failure_study.ObservationInput,
        str(config.get("observation_input", "rectified_board")),
    )
    rows_by_clip = failure_study._load_rows_by_clip(observation_input)
    clip_key = str(entry.get("clip_path") or "")
    clip_rows = rows_by_clip.get(clip_key)
    if clip_rows is None:
        raise FileNotFoundError(f"Clip {clip_key} not found for failure-study context")

    anchor_position = _find_anchor_position(
        clip_rows,
        annotation_id=str(entry.get("annotation_id") or ""),
        frame_index=int(entry.get("frame_index", 0)),
    )
    start_position = max(0, anchor_position - context_frames)
    window_rows = clip_rows[start_position : anchor_position + 1]

    clip_cache: dict[Path, dict[str, Any]] = {}
    frames = [
        {
            "annotation_id": row.annotation_id,
            "frame_index": int(getattr(row, "frame_index", 0)),
            "relative_offset": int(getattr(row, "frame_index", 0)) - int(entry["frame_index"]),
            "is_anchor": row.annotation_id == entry["annotation_id"],
            "image_data_url": _image_data_url(
                failure_study._load_row_image(
                    row,
                    observation_input=observation_input,
                    clip_cache=clip_cache,
                ),
                max_side=image_max_side,
            ),
        }
        for row in window_rows
    ]

    return {
        "study_path": _relative_to_project(bundle["study_dir"]),
        "selected_index": selected_index,
        "context_frames": context_frames,
        "observation_input": observation_input,
        "entry": entry,
        "anchor_panel_data_url": _file_data_url(entry.get("image_path")),
        "frames": frames,
    }


def update_failure_study_entry(
    study_path: str,
    *,
    selected_index: int,
    final_bucket: str | None,
    notes: str | None,
) -> dict[str, Any]:
    if selected_index <= 0:
        raise ValueError(f"selected_index must be > 0, got {selected_index}")

    bundle = _load_bundle(_resolve_study_dir(study_path))
    fieldnames, rows = _load_csv_rows(bundle["manual_buckets_csv_path"])

    updated_row: dict[str, str] | None = None
    for row in rows:
        if int(row.get("selected_index", "0") or 0) != selected_index:
            continue
        row["final_bucket"] = (final_bucket or "").strip()
        row["notes"] = notes or ""
        updated_row = row
        break

    if updated_row is None:
        raise FileNotFoundError(
            f"Failure-study entry {selected_index} not found in {bundle['manual_buckets_csv_path']}"
        )

    with bundle["manual_buckets_csv_path"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "selected_index": selected_index,
        "final_bucket": updated_row.get("final_bucket", ""),
        "notes": updated_row.get("notes", ""),
    }


def _candidate_summary_paths() -> list[Path]:
    if not _OUTPUTS_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in _OUTPUTS_DIR.rglob("summary.json")
            if "failure_study" in str(path.parent)
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _resolve_study_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.name == "summary.json":
        candidate = candidate.parent
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Failure-study directory not found: {path}")
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"Failure-study path must be inside project root: {path}")
    return resolved


def _load_bundle(study_dir: Path) -> dict[str, Any]:
    summary_path = study_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Failure-study summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text())

    manifest_path = _resolve_bundle_path(
        summary.get("manifest"),
        fallback=study_dir / "manifest.json",
    )
    manual_buckets_csv_path = _resolve_bundle_path(
        summary.get("manual_buckets_csv"),
        fallback=study_dir / "manual_buckets.csv",
    )
    report_path = _resolve_optional_bundle_path(summary.get("report_path"))

    if not manifest_path.exists():
        raise FileNotFoundError(f"Failure-study manifest not found: {manifest_path}")
    if not manual_buckets_csv_path.exists():
        raise FileNotFoundError(
            f"Failure-study bucket CSV not found: {manual_buckets_csv_path}"
        )

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, list):
        raise ValueError(f"Failure-study manifest must be a list: {manifest_path}")

    return {
        "study_dir": study_dir,
        "summary_path": summary_path,
        "summary": summary,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manual_buckets_csv_path": manual_buckets_csv_path,
        "report_path": report_path,
    }


def _resolve_bundle_path(value: Any, *, fallback: Path) -> Path:
    resolved = _resolve_optional_bundle_path(value)
    return fallback if resolved is None else resolved


def _resolve_optional_bundle_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"Failure-study path must be inside project root: {value}")
    return resolved


def _load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Failure-study CSV is missing headers: {path}")
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    return fieldnames, rows


def _merged_entries(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    _fieldnames, csv_rows = _load_csv_rows(bundle["manual_buckets_csv_path"])
    csv_by_selected_index = {
        int(row.get("selected_index", "0") or 0): row for row in csv_rows
    }

    entries: list[dict[str, Any]] = []
    for raw_entry in bundle["manifest"]:
        entry = dict(raw_entry)
        selected_index = int(entry.get("selected_index", 0))
        csv_row = csv_by_selected_index.get(selected_index, {})
        entry["selected_index"] = selected_index
        entry["frame_index"] = int(entry.get("frame_index", 0))
        entry["decoded_error_count"] = int(entry.get("decoded_error_count", 0))
        entry["stateless_error_count"] = int(entry.get("stateless_error_count", 0))
        entry["final_bucket"] = csv_row.get("final_bucket", "")
        entry["notes"] = csv_row.get("notes", "")
        entry["image_path"] = entry.get("image_path")
        legal = entry.get("legal_from_previous_decoded")
        if not isinstance(legal, dict):
            legal = {}
        entry["best_legal_matches_gt"] = bool(legal.get("best_legal_matches_gt"))
        entry["best_legal_move_uci"] = legal.get("best_legal_move_uci")
        entry["gt_legal_rank"] = legal.get("gt_legal_rank")
        entries.append(entry)

    entries.sort(key=lambda item: (int(item["selected_index"]), int(item["frame_index"])))
    return entries


def _load_report_metrics(report_path: Path | None) -> dict[str, Any] | None:
    if report_path is None or not report_path.exists():
        return None
    payload = json.loads(report_path.read_text())
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {
        "board_exact_match": metrics.get("board_exact_match"),
        "non_empty_accuracy": metrics.get("non_empty_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "accuracy": metrics.get("accuracy"),
        "move_detection_recall": payload.get("move_detection_recall"),
        "static_frame_false_change_rate": payload.get("static_frame_false_change_rate"),
    }


def _bucket_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        bucket = str(entry.get("final_bucket") or "").strip() or "(untagged)"
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _find_anchor_position(
    clip_rows: list[Any],
    *,
    annotation_id: str,
    frame_index: int,
) -> int:
    for index, row in enumerate(clip_rows):
        if row.annotation_id == annotation_id:
            return index
    for index, row in enumerate(clip_rows):
        if int(getattr(row, "frame_index", -1)) == frame_index:
            return index
    raise FileNotFoundError(
        f"Failure-study anchor {annotation_id or frame_index} not found in clip rows"
    )


def _image_data_url(image_bgr: np.ndarray, *, max_side: int) -> str:
    resized = _resize_image(image_bgr, max_side=max_side)
    success, encoded = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
    if not success:
        raise ValueError("Failed to encode failure-study context image")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _file_data_url(path_value: Any) -> str | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    suffix = resolved.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix)
    if mime is None:
        return None
    payload = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _resize_image(image_bgr: np.ndarray, *, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return image_bgr
    scale = max_side / float(largest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _isoformat_mtime(path: Path) -> str | None:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))
    except OSError:
        return None


def _relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)
