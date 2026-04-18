"""Service layer for browsing physical board failure-study bundles."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2

from pipeline.paths import PROJECT_ROOT
from pipeline.physical.board_probe.failure_study import BUCKETS
from pipeline.physical.shared.annotation_rows import (
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)

_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def list_failure_studies() -> list[dict[str, Any]]:
    studies: list[dict[str, Any]] = []
    for summary_path in _candidate_summary_paths():
        try:
            bundle = _load_bundle(summary_path.parent)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, csv.Error):
            continue

        summary = bundle["summary"]
        config = summary.get("config") if isinstance(summary, dict) else None
        if not isinstance(config, dict):
            continue

        selected_episodes = int(
            summary.get("selected_episodes", summary.get("selected_failures", 0))
        )
        total_episodes = int(summary.get("total_episodes", summary.get("total_failures", 0)))
        studies.append(
            {
                "path": _relative_to_project(bundle["study_dir"]),
                "label": bundle["study_dir"].name,
                "modified_at": _isoformat_mtime(bundle["summary_path"]),
                "selected_episodes": selected_episodes,
                "total_episodes": total_episodes,
                "observation_input": config.get("observation_input"),
                "tracker_mode": config.get("tracker_mode"),
                "report_path": summary.get("report_path"),
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
    report_metrics = _load_report_metrics(bundle.get("report_path"))
    entries = _merge_annotations(bundle)
    return {
        "path": _relative_to_project(bundle["study_dir"]),
        "label": bundle["study_dir"].name,
        "modified_at": _isoformat_mtime(bundle["summary_path"]),
        "summary": bundle["summary"],
        "report_metrics": report_metrics,
        "entries": entries,
        "bucket_options": list(BUCKETS),
        "bucket_counts": _bucket_counts(entries),
    }


def resolve_image_path(study_path: str, image_path: str) -> Path:
    _load_bundle(_resolve_study_dir(study_path))
    target = (PROJECT_ROOT / image_path).resolve()
    try:
        target.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise PermissionError(f"Image path escaped project root: {image_path}") from error
    if not target.is_file():
        raise FileNotFoundError(f"Failure-study image not found: {image_path}")
    return target


def update_failure_study_entry(
    study_path: str,
    *,
    episode_id: str,
    final_bucket: str | None,
    notes: str | None,
) -> dict[str, Any]:
    if not episode_id:
        raise ValueError("episode_id is required")
    if final_bucket is not None and final_bucket.strip() and final_bucket not in BUCKETS:
        raise ValueError(f"Unknown bucket: {final_bucket}")

    bundle = _load_bundle(_resolve_study_dir(study_path))
    fieldnames, rows = _load_csv_rows(bundle["manual_buckets_csv_path"])

    updated_row: dict[str, str] | None = None
    updated_at = datetime.now(tz=timezone.utc).isoformat()
    for row in rows:
        if row.get("episode_id") != episode_id:
            continue
        row["final_bucket"] = (final_bucket or "").strip()
        row["notes"] = notes or ""
        if "bucket_updated_at" in fieldnames:
            row["bucket_updated_at"] = updated_at
        updated_row = row
        break

    if updated_row is None:
        raise FileNotFoundError(
            f"Failure-study episode {episode_id} not found in {bundle['manual_buckets_csv_path']}"
        )

    with bundle["manual_buckets_csv_path"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "episode_id": episode_id,
        "final_bucket": updated_row.get("final_bucket", ""),
        "notes": updated_row.get("notes", ""),
        "updated_at": updated_row.get("bucket_updated_at") or updated_at,
    }


def export_manual_buckets_csv(study_path: str) -> str:
    bundle = _load_bundle(_resolve_study_dir(study_path))
    return bundle["manual_buckets_csv_path"].read_text()


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


def _merge_annotations(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    _fieldnames, csv_rows = _load_csv_rows(bundle["manual_buckets_csv_path"])
    csv_by_episode_id = {
        row.get("episode_id", ""): row for row in csv_rows if row.get("episode_id")
    }
    raw_snapshot_paths: dict[str, str | None] = {}
    clip_cache: dict[Path, dict[str, Any]] = {}

    entries: list[dict[str, Any]] = []
    for raw_entry in bundle["manifest"]:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        selected_index = int(entry.get("selected_index", 0) or 0)
        episode_id = str(entry.get("episode_id") or f"entry-{selected_index:03d}")
        csv_row = csv_by_episode_id.get(episode_id, {})

        if isinstance(entry.get("failing_frame"), dict):
            entry["failing_frame"] = _normalize_frame(
                entry.get("failing_frame") or {},
                study_dir=bundle["study_dir"],
                raw_snapshot_paths=raw_snapshot_paths,
                clip_cache=clip_cache,
            )
            entry["preceding_frames"] = [
                _normalize_frame(
                    frame,
                    study_dir=bundle["study_dir"],
                    raw_snapshot_paths=raw_snapshot_paths,
                    clip_cache=clip_cache,
                )
                for frame in entry.get("preceding_frames") or []
            ]
            entry["first_frame_index"] = int(entry.get("first_frame_index", 0) or 0)
            entry["last_frame_index"] = int(entry.get("last_frame_index", 0) or 0)
            entry["length"] = int(entry.get("length", 0) or 0)
            entry["preceding_frame_count"] = int(entry.get("preceding_frame_count", 0) or 0)
            entry["suggested_bucket"] = str(entry.get("suggested_bucket") or "")
        else:
            normalized = _normalize_frame(
                entry,
                study_dir=bundle["study_dir"],
                raw_snapshot_paths=raw_snapshot_paths,
                clip_cache=clip_cache,
            )
            entry["failing_frame"] = {**normalized, "is_failing_frame": True}
            entry["preceding_frames"] = []
            entry["first_frame_index"] = normalized["frame_index"]
            entry["last_frame_index"] = normalized["frame_index"]
            entry["length"] = 1
            entry["preceding_frame_count"] = 0
            entry["suggested_bucket"] = str(entry.get("suggested_bucket") or "")

        entry["episode_id"] = episode_id
        entry["selected_index"] = selected_index
        entry["source_video_id"] = str(entry.get("source_video_id") or "")
        entry["clip_path"] = str(entry.get("clip_path") or "")
        entry["clip_filename"] = str(entry.get("clip_filename") or "")
        entry["final_bucket"] = csv_row.get("final_bucket", "")
        entry["notes"] = csv_row.get("notes", "")
        entry["bucket_updated_at"] = csv_row.get("bucket_updated_at")
        entries.append(entry)

    entries.sort(key=lambda item: (int(item["selected_index"]), item["episode_id"]))
    return entries


@lru_cache(maxsize=1)
def _annotation_rows_by_id() -> dict[str, Any]:
    rows_by_id: dict[str, Any] = {}
    for split in ("val", "train"):
        annotation_root = PROJECT_ROOT / "data" / "physical" / split
        annotations_path = annotation_root / "board_annotations.jsonl"
        if not annotations_path.exists():
            continue
        for row in load_annotated_oblique_rows(annotation_root):
            rows_by_id[row.annotation_id] = row
    return rows_by_id


def _raw_snapshot_path(
    *,
    study_dir: Path,
    annotation_id: str,
    raw_snapshot_paths: dict[str, str | None],
    clip_cache: dict[Path, dict[str, Any]],
) -> str | None:
    if not annotation_id:
        return None
    cached = raw_snapshot_paths.get(annotation_id)
    if annotation_id in raw_snapshot_paths:
        return cached

    try:
        row = _annotation_rows_by_id().get(annotation_id)
        if row is None:
            raw_snapshot_paths[annotation_id] = None
            return None

        snapshots_dir = study_dir / "viewer_cache" / "raw_snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshots_dir / f"{annotation_id}.png"
        if not snapshot_path.exists():
            frame_bgr = _load_clip_frame_bgr(row, clip_cache=clip_cache)
            encoded, buffer = cv2.imencode(".png", frame_bgr)
            if not encoded:
                raise ValueError(f"Failed to encode raw snapshot for {annotation_id}")
            snapshot_path.write_bytes(buffer.tobytes())
        raw_snapshot_paths[annotation_id] = _relative_to_project(snapshot_path)
    except Exception:
        raw_snapshot_paths[annotation_id] = None

    return raw_snapshot_paths[annotation_id]


def _normalize_frame(
    frame: dict[str, Any],
    *,
    study_dir: Path,
    raw_snapshot_paths: dict[str, str | None],
    clip_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    legal = frame.get("legal_from_previous_decoded")
    if not isinstance(legal, dict):
        legal = None
    square_diagnostics = frame.get("square_diagnostics")
    if not isinstance(square_diagnostics, list):
        square_diagnostics = []
    annotation_id = str(frame.get("annotation_id") or "")
    processed_image_path = frame.get("board_path")
    if not isinstance(processed_image_path, str) or not processed_image_path.strip():
        processed_image_path = None
    return {
        **frame,
        "annotation_id": annotation_id,
        "clip_path": frame.get("clip_path"),
        "clip_filename": str(frame.get("clip_filename") or ""),
        "frame_index": int(frame.get("frame_index", 0) or 0),
        "board_path": frame.get("board_path"),
        "processed_image_path": processed_image_path,
        "raw_image_path": _raw_snapshot_path(
            study_dir=study_dir,
            annotation_id=annotation_id,
            raw_snapshot_paths=raw_snapshot_paths,
            clip_cache=clip_cache,
        ),
        "source_video_id": frame.get("source_video_id"),
        "gt_fen": str(frame.get("gt_fen") or ""),
        "decoded_fen": str(frame.get("decoded_fen") or ""),
        "decoded_full_fen": str(frame.get("decoded_full_fen") or ""),
        "stateless_fen": str(frame.get("stateless_fen") or ""),
        "decoded_move_uci": frame.get("decoded_move_uci"),
        "decoded_move_score": frame.get("decoded_move_score"),
        "decoded_stay_score": frame.get("decoded_stay_score"),
        "decoded_error_count": int(frame.get("decoded_error_count", 0) or 0),
        "stateless_error_count": int(frame.get("stateless_error_count", 0) or 0),
        "decoded_matches_previous_gt": bool(frame.get("decoded_matches_previous_gt")),
        "decoded_matches_next_gt": bool(frame.get("decoded_matches_next_gt")),
        "stateless_error_squares": frame.get("stateless_error_squares") or [],
        "decoded_error_squares": frame.get("decoded_error_squares") or [],
        "decoded_changed_squares": frame.get("decoded_changed_squares") or [],
        "stateless_changed_squares": frame.get("stateless_changed_squares") or [],
        "gt_changed_squares": frame.get("gt_changed_squares") or [],
        "legal_from_previous_decoded": legal,
        "square_diagnostics": square_diagnostics,
        "decoded_square_confidences": frame.get("decoded_square_confidences") or [],
        "stateless_square_confidences": frame.get("stateless_square_confidences") or [],
        "suggested_bucket": frame.get("suggested_bucket"),
        "image_path": str(frame.get("image_path") or ""),
        "offset_from_failure": int(frame.get("offset_from_failure", 0) or 0),
        "is_failing_frame": bool(frame.get("is_failing_frame")),
    }


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
