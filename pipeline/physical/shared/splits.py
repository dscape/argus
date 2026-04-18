"""Split-aware storage helpers for physical-board annotations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

PhysicalAnnotationSplit = Literal["train", "val"]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _PROJECT_ROOT / "data" / "physical"
_SOURCE_VIDEO_SPLITS_PATH = _DATA_ROOT / "source_video_splits.json"
_CANONICAL_ROOTS: dict[PhysicalAnnotationSplit, Path] = {
    "train": _DATA_ROOT / "train",
    "val": _DATA_ROOT / "val",
}
_LEGACY_ROOTS: dict[PhysicalAnnotationSplit, Path] = {
    "train": _DATA_ROOT / "train_manual",
    "val": _DATA_ROOT / "eval",
}
_LEGACY_SPLIT_NAMES = {
    "train": "train_manual",
    "val": "eval_holdout",
}
_VALID_SPLITS: tuple[PhysicalAnnotationSplit, ...] = ("train", "val")
_VAL_SPLIT_FRACTION = 0.2


def normalize_split(split: str) -> PhysicalAnnotationSplit:
    normalized = str(split).strip().lower()
    if normalized not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    return normalized  # type: ignore[return-value]


def split_root(split: str) -> Path:
    return _CANONICAL_ROOTS[normalize_split(split)]


def source_video_splits_path() -> Path:
    return _SOURCE_VIDEO_SPLITS_PATH


def ensure_annotation_layout_migrated() -> None:
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for split in _VALID_SPLITS:
        _migrate_split_root(split)
    _write_source_video_splits_manifest()


def load_source_video_splits() -> dict[str, PhysicalAnnotationSplit]:
    ensure_annotation_layout_migrated()
    return _load_existing_source_video_splits()


def get_source_video_ids_for_split(split: str) -> list[str]:
    normalized_split = normalize_split(split)
    return sorted(
        source_video_id
        for source_video_id, assigned_split in load_source_video_splits().items()
        if assigned_split == normalized_split
    )


def ensure_source_video_splits_assigned(
    source_video_ids: Iterable[str | None],
) -> dict[str, PhysicalAnnotationSplit]:
    assignments = load_source_video_splits()
    missing_source_video_ids = sorted({
        str(source_video_id)
        for source_video_id in source_video_ids
        if source_video_id and str(source_video_id) not in assignments
    })
    if not missing_source_video_ids:
        return assignments

    for source_video_id in missing_source_video_ids:
        assignments[source_video_id] = _auto_assign_source_video_split(source_video_id)
    _save_source_video_splits(assignments)
    return assignments


def assign_source_video_split(
    source_video_id: str | None,
    split: str,
) -> PhysicalAnnotationSplit | None:
    if not source_video_id:
        return None

    normalized_split = normalize_split(split)
    assignments = load_source_video_splits()
    existing_split = assignments.get(source_video_id)
    if existing_split is not None and existing_split != normalized_split:
        raise ValueError(
            f"Source video {source_video_id} is assigned to {existing_split}, "
            f"not {normalized_split}"
        )
    if existing_split == normalized_split:
        return normalized_split

    assignments[source_video_id] = normalized_split
    _save_source_video_splits(assignments)
    return normalized_split


def _auto_assign_source_video_split(source_video_id: str) -> PhysicalAnnotationSplit:
    if _stable_assignment_fraction(source_video_id) < _VAL_SPLIT_FRACTION:
        return "val"
    return "train"


def _stable_assignment_fraction(source_video_id: str) -> float:
    digest = hashlib.sha256(source_video_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(1 << 64)


def _migrate_split_root(split: PhysicalAnnotationSplit) -> None:
    canonical_root = _CANONICAL_ROOTS[split]
    legacy_root = _LEGACY_ROOTS[split]
    if not legacy_root.exists():
        canonical_root.mkdir(parents=True, exist_ok=True)
        return

    canonical_root.mkdir(parents=True, exist_ok=True)
    _merge_directory_contents(legacy_root / "boards", canonical_root / "boards")
    _merge_directory_contents(legacy_root / "squares", canonical_root / "squares")

    _merge_manifest(
        legacy_root / "board_annotations.jsonl",
        canonical_root / "board_annotations.jsonl",
        key_fields=("annotation_id",),
        split=split,
        path_fields=("rectified_board_path",),
    )
    _merge_manifest(
        legacy_root / "square_manifest.jsonl",
        canonical_root / "square_manifest.jsonl",
        key_fields=("annotation_id", "square_index"),
        split=split,
        path_fields=("crop_path",),
    )

    shutil.rmtree(legacy_root)


def _merge_directory_contents(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = destination / path.name
        if target.exists():
            continue
        shutil.move(str(path), str(target))


def _merge_manifest(
    source_path: Path,
    destination_path: Path,
    *,
    key_fields: tuple[str, ...],
    split: PhysicalAnnotationSplit,
    path_fields: tuple[str, ...],
) -> None:
    source_rows = [
        _rewrite_manifest_row(row, split=split, path_fields=path_fields)
        for row in _load_jsonl(source_path)
    ]
    destination_rows = [
        _rewrite_manifest_row(row, split=split, path_fields=path_fields)
        for row in _load_jsonl(destination_path)
    ]
    if not source_rows and not destination_rows:
        return

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in source_rows:
        merged[_row_key(row, key_fields)] = row
    for row in destination_rows:
        merged[_row_key(row, key_fields)] = row

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = list(merged.values())
    ordered_rows.sort(
        key=lambda row: (
            str(row.get("clip_path", "")),
            int(row.get("frame_index", -1)),
            str(row.get("annotation_id", "")),
            int(row.get("square_index", -1)),
        )
    )
    _write_jsonl(destination_path, ordered_rows)


def _rewrite_manifest_row(
    row: dict[str, Any],
    *,
    split: PhysicalAnnotationSplit,
    path_fields: tuple[str, ...],
) -> dict[str, Any]:
    rewritten = dict(row)
    for field in path_fields:
        raw_value = rewritten.get(field)
        if not isinstance(raw_value, str):
            continue
        rewritten[field] = _rewrite_relative_path(raw_value, split=split)

    raw_split = rewritten.get("split")
    if isinstance(raw_split, str) and raw_split == _LEGACY_SPLIT_NAMES[split]:
        rewritten["split"] = split
    return rewritten


def _rewrite_relative_path(relative_path: str, *, split: PhysicalAnnotationSplit) -> str:
    old_prefix = f"data/physical/{_LEGACY_ROOTS[split].name}/"
    new_prefix = f"data/physical/{split}/"
    if relative_path.startswith(old_prefix):
        return new_prefix + relative_path[len(old_prefix):]
    return relative_path


def _row_key(row: dict[str, Any], key_fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in key_fields)


def _write_source_video_splits_manifest() -> None:
    existing_assignments = _load_existing_source_video_splits()

    inferred_assignments = dict(existing_assignments)
    for split in _VALID_SPLITS:
        manifest_path = split_root(split) / "board_annotations.jsonl"
        for source_video_id in _source_video_ids_from_manifest(manifest_path):
            existing_split = inferred_assignments.get(source_video_id)
            if existing_split is not None and existing_split != split:
                raise ValueError(
                    f"Source video {source_video_id} is assigned to both "
                    f"{existing_split} and {split}"
                )
            inferred_assignments[source_video_id] = split

    _save_source_video_splits(inferred_assignments)


def _save_source_video_splits(assignments: dict[str, PhysicalAnnotationSplit]) -> None:
    payload = {
        "version": 1,
        "source_video_splits": {
            source_video_id: assignments[source_video_id]
            for source_video_id in sorted(assignments)
        },
    }
    _atomic_write_text(
        _SOURCE_VIDEO_SPLITS_PATH,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _load_existing_source_video_splits() -> dict[str, PhysicalAnnotationSplit]:
    if not _SOURCE_VIDEO_SPLITS_PATH.exists():
        return {}

    raw_payload = _SOURCE_VIDEO_SPLITS_PATH.read_text()
    if not raw_payload.strip():
        return {}

    payload = json.loads(raw_payload)
    if isinstance(payload, dict) and isinstance(payload.get("source_video_splits"), dict):
        raw_assignments = payload["source_video_splits"]
    elif isinstance(payload, dict):
        raw_assignments = payload
    else:
        raise ValueError(
            f"Invalid source video split manifest: {_SOURCE_VIDEO_SPLITS_PATH}"
        )

    assignments: dict[str, PhysicalAnnotationSplit] = {}
    for source_video_id, split in raw_assignments.items():
        if not isinstance(source_video_id, str) or not isinstance(split, str):
            continue
        assignments[source_video_id] = normalize_split(split)
    return assignments


def _source_video_ids_from_manifest(path: Path) -> set[str]:
    return {
        str(row["source_video_id"])
        for row in _load_jsonl(path)
        if row.get("source_video_id")
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(f"{content}\n" if content else "")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    temp_path.replace(path)
