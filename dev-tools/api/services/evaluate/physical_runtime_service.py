"""Service layer for visualizing and evaluating the physical runtime reader."""

from __future__ import annotations

import base64
import json
import random
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from pipeline.db.connection import get_conn
from pipeline.paths import PROJECT_ROOT
from pipeline.physical.board_data import PhysicalEvalBoardDataset, PhysicalEvalBoardRow
from pipeline.physical.runtime_visualization import (
    VisualizedRuntimeFrame,
    _collect_visualized_frames,
    _collect_visualized_frames_for_indices,
    _group_rows_by_clip,
    _select_clip_path,
    render_contact_sheet,
    render_visualized_runtime_frame,
)
from pipeline.shared import SQUARE_CLASS_NAMES

_SESSION_IMAGES_DIR = PROJECT_ROOT / "data" / "eval_sessions"
_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "physical"
_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
_EMPTY_CLASS_ID = 0


def render_runtime_visualization(
    *,
    clip_path: str | None,
    frame_start: int,
    frame_count: int,
    panel_size: int = 240,
    device: str = "cpu",
    model_path: str | None = None,
) -> dict[str, Any]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    if panel_size <= 0:
        raise ValueError(f"panel_size must be > 0, got {panel_size}")

    dataset = PhysicalEvalBoardDataset()
    rows_by_clip = _group_rows_by_clip(dataset.rows)
    selected_clip_path = _select_clip_path(rows_by_clip, clip_path=clip_path)
    clip_rows = rows_by_clip[selected_clip_path]
    collect_kwargs: dict[str, Any] = {
        "frame_start": frame_start,
        "frame_count": frame_count,
        "device": device,
    }
    if model_path is not None:
        collect_kwargs["weights_path"] = model_path
    visualized_frames = _collect_visualized_frames(
        clip_rows,
        **collect_kwargs,
    )

    frame_images = [
        render_visualized_runtime_frame(frame, panel_size=panel_size) for frame in visualized_frames
    ]
    contact_sheet = render_contact_sheet(
        frame_images,
        clip_path=selected_clip_path,
        frame_start=frame_start,
        frame_count=len(visualized_frames),
    )

    return {
        "clip_path": selected_clip_path,
        "frame_start": frame_start,
        "frame_count": len(visualized_frames),
        "available_frame_count": len(clip_rows),
        "contact_sheet_b64": _image_to_base64(contact_sheet),
        "frames": [
            {
                "annotation_id": frame.annotation_id,
                "board_path": frame.board_path,
                "frame_index": frame.frame_index,
                "gt_change_count": frame.gt_change_count,
                "stateless_change_count": frame.stateless_change_count,
                "stateless_error_count": frame.stateless_error_count,
                "stateless_mean_confidence": round(frame.stateless_mean_confidence, 4),
                "temporal_change_count": frame.temporal_change_count,
                "temporal_error_count": frame.temporal_error_count,
                "temporal_mean_confidence": round(frame.temporal_mean_confidence, 4),
                "image_b64": _image_to_base64(frame_image),
            }
            for frame, frame_image in zip(visualized_frames, frame_images)
        ],
    }


def sample_runtime_frames(
    limit: int,
    exclude_annotation_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError(f"limit must be > 0, got {limit}")

    excluded = set(exclude_annotation_ids or [])
    rows = [row for row in PhysicalEvalBoardDataset().rows if row.annotation_id not in excluded]
    random.shuffle(rows)
    sampled = rows[: min(limit, len(rows))]
    return [
        {
            "annotation_id": row.annotation_id,
            "clip_path": row.clip_path,
            "clip_filename": _clip_filename(row),
            "frame_index": row.frame_index,
        }
        for row in sampled
    ]


def list_runtime_models() -> list[dict[str, Any]]:
    candidates: dict[str, Path] = {}

    for path in sorted(_WEIGHTS_DIR.glob("*.pt")):
        if path.is_file():
            candidates[_relative_to_project(path)] = path

    if _OUTPUTS_DIR.exists():
        patterns = ("**/board_probe.pt", "**/linear_probe.pt", "**/physical_board_probe_ensemble*.pt")
        for pattern in patterns:
            for path in _OUTPUTS_DIR.glob(pattern):
                if path.is_file():
                    candidates[_relative_to_project(path)] = path

    default_version = _default_runtime_version()
    models: list[dict[str, Any]] = []
    for relative_path, path in candidates.items():
        source = "weights" if path.is_relative_to(_WEIGHTS_DIR) else "outputs"
        try:
            modified_ts = path.stat().st_mtime
        except OSError:
            modified_ts = 0.0
        models.append(
            {
                "path": relative_path,
                "label": _runtime_model_label(path, default_version=default_version),
                "source": source,
                "is_default": path == _WEIGHTS_DIR / "best.pt",
                "modified_at": _isoformat_mtime(path),
                "_modified_ts": modified_ts,
            }
        )

    models.sort(
        key=lambda model: (
            0 if model["is_default"] else 1,
            0 if model["source"] == "weights" else 1,
            -float(model["_modified_ts"]),
            str(model["label"]),
        )
    )
    for model in models:
        model.pop("_modified_ts", None)
    return models


def inspect_runtime_frame(
    *,
    annotation_id: str,
    panel_size: int = 240,
    device: str = "cpu",
    model_path: str | None = None,
) -> dict[str, Any]:
    results = inspect_runtime_frames(
        annotation_ids=[annotation_id],
        panel_size=panel_size,
        device=device,
        model_path=model_path,
    )
    if len(results) != 1:
        raise ValueError(f"Expected exactly one result for {annotation_id}, got {len(results)}")
    return results[0]



def inspect_runtime_frames(
    *,
    annotation_ids: list[str],
    panel_size: int = 240,
    device: str = "cpu",
    model_path: str | None = None,
) -> list[dict[str, Any]]:
    if panel_size <= 0:
        raise ValueError(f"panel_size must be > 0, got {panel_size}")
    if not annotation_ids:
        return []

    rows_by_clip = _rows_by_clip()
    selected_rows_by_annotation_id = _find_rows_by_annotation_ids(annotation_ids, rows_by_clip)
    selected_rows_by_clip: dict[str, list[PhysicalEvalBoardRow]] = {}
    for annotation_id in annotation_ids:
        selected_row = selected_rows_by_annotation_id[annotation_id]
        clip_key = selected_row.clip_path or selected_row.annotation_id
        selected_rows_by_clip.setdefault(clip_key, []).append(selected_row)

    results_by_annotation_id: dict[str, dict[str, Any]] = {}
    for clip_key, clip_selected_rows in selected_rows_by_clip.items():
        clip_rows = rows_by_clip[clip_key]
        started_at = time.perf_counter()
        collect_kwargs: dict[str, Any] = {
            "selected_frame_indices": {
                int(row.frame_index)
                for row in clip_selected_rows
                if row.frame_index is not None
            },
            "device": device,
        }
        if model_path is not None:
            collect_kwargs["weights_path"] = model_path
        visualized_frames = _collect_visualized_frames_for_indices(
            clip_rows,
            **collect_kwargs,
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 1)
        elapsed_ms_per_frame = round(elapsed_ms / len(visualized_frames), 1)

        visualized_by_annotation_id = {
            frame.annotation_id: frame for frame in visualized_frames
        }
        for selected_row in clip_selected_rows:
            frame = visualized_by_annotation_id.get(selected_row.annotation_id)
            if frame is None:
                raise ValueError(
                    "Expected visualized frame for "
                    f"{selected_row.annotation_id}, got {len(visualized_frames)} frames"
                )
            results_by_annotation_id[selected_row.annotation_id] = _runtime_result_from_frame(
                selected_row,
                frame,
                elapsed_ms=elapsed_ms_per_frame,
            )

    return [results_by_annotation_id[annotation_id] for annotation_id in annotation_ids]


def save_physical_runtime_eval(
    square_accuracy: float,
    non_empty_accuracy: float | None,
    exact_match_rate: float | None,
    sample_size: int,
    elapsed_ms_avg: float | None = None,
    images_per_minute: int | None = None,
    stateless_square_accuracy: float | None = None,
    stateless_non_empty_accuracy: float | None = None,
    stateless_exact_match_rate: float | None = None,
    notes: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    per_class_data: dict[str, Any] = {}
    if non_empty_accuracy is not None:
        per_class_data["non_empty_accuracy"] = round(non_empty_accuracy, 4)
    if exact_match_rate is not None:
        per_class_data["exact_match_rate"] = round(exact_match_rate, 4)
    if elapsed_ms_avg is not None:
        per_class_data["elapsed_ms_avg"] = round(elapsed_ms_avg, 3)
    if images_per_minute is not None:
        per_class_data["images_per_minute"] = images_per_minute
    if stateless_square_accuracy is not None:
        per_class_data["stateless_square_accuracy"] = round(stateless_square_accuracy, 4)
    if stateless_non_empty_accuracy is not None:
        per_class_data["stateless_non_empty_accuracy"] = round(stateless_non_empty_accuracy, 4)
    if stateless_exact_match_rate is not None:
        per_class_data["stateless_exact_match_rate"] = round(stateless_exact_match_rate, 4)
    if model_path is not None:
        per_class_data["model_path"] = model_path
    if notes is not None:
        per_class_data["model_label"] = notes

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_evaluations
                   (model_name, sample_size, accuracy, notes, per_class)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id, evaluated_at""",
                (
                    "physical",
                    sample_size,
                    round(square_accuracy, 4),
                    notes,
                    json.dumps(per_class_data) if per_class_data else None,
                ),
            )
            row = cur.fetchone()
            conn.commit()
    return {"id": row[0], "evaluated_at": str(row[1])}


def create_physical_runtime_session(
    results: list[dict[str, Any]],
    square_accuracy: float | None,
    non_empty_accuracy: float | None,
    exact_match_rate: float | None,
    sample_size: int,
    pin_state: dict | None = None,
    evaluation_id: int | None = None,
) -> dict[str, Any]:
    session_id = uuid.uuid4().hex[:12]
    lightweight: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        entry = dict(result)
        annotation_id = str(entry.get("annotation_id", f"frame_{index}"))
        if entry.get("thumbnail_b64"):
            entry["thumbnail_filename"] = _save_session_image(
                session_id,
                f"{index:03d}_{_safe_filename_stem(annotation_id)}_thumb",
                str(entry["thumbnail_b64"]),
            )
            entry.pop("thumbnail_b64", None)
        if entry.get("image_b64"):
            entry["image_filename"] = _save_session_image(
                session_id,
                f"{index:03d}_{_safe_filename_stem(annotation_id)}",
                str(entry["image_b64"]),
            )
            entry.pop("image_b64", None)
        lightweight.append(entry)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO physical_runtime_sessions
                   (id, sample_size, square_accuracy, non_empty_accuracy,
                    exact_match_rate, results, pin_state, evaluation_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    session_id,
                    sample_size,
                    square_accuracy,
                    non_empty_accuracy,
                    exact_match_rate,
                    json.dumps(lightweight),
                    json.dumps(pin_state or {}),
                    evaluation_id,
                ),
            )
            conn.commit()
    return {"session_id": session_id}


def get_physical_runtime_session(session_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT s.id, s.created_at, s.sample_size, s.square_accuracy,
                          s.non_empty_accuracy, s.exact_match_rate,
                          s.results, s.pin_state, s.evaluation_id,
                          e.notes, e.per_class
                   FROM physical_runtime_sessions s
                   LEFT JOIN model_evaluations e ON e.id = s.evaluation_id
                   WHERE s.id = %s""",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None

    results = row[6] if isinstance(row[6], list) else json.loads(row[6])
    pin_state = row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}")
    evaluation_per_class = row[10] if isinstance(row[10], dict) else json.loads(row[10] or "{}")
    model_path = _evaluation_model_path(evaluation_per_class, row[9])
    model_label = _evaluation_model_label(evaluation_per_class, row[9], model_path)

    return {
        "id": row[0],
        "created_at": str(row[1]),
        "sample_size": row[2],
        "square_accuracy": row[3],
        "non_empty_accuracy": row[4],
        "exact_match_rate": row[5],
        "results": results,
        "pin_state": pin_state,
        "evaluation_id": row[8],
        "model_label": model_label,
        "model_path": model_path,
    }


def list_physical_runtime_sessions(limit: int = 20) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT s.id, s.created_at, s.sample_size, s.square_accuracy,
                          s.non_empty_accuracy, s.exact_match_rate,
                          e.notes, e.per_class
                   FROM physical_runtime_sessions s
                   LEFT JOIN model_evaluations e ON e.id = s.evaluation_id
                   ORDER BY s.created_at DESC LIMIT %s""",
                (limit,),
            )
            return [
                {
                    "id": row[0],
                    "created_at": str(row[1]),
                    "sample_size": row[2],
                    "square_accuracy": row[3],
                    "non_empty_accuracy": row[4],
                    "exact_match_rate": row[5],
                    "model_label": _evaluation_model_label(
                        row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}"),
                        row[6],
                        _evaluation_model_path(
                            row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}"),
                            row[6],
                        ),
                    ),
                    "model_path": _evaluation_model_path(
                        row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}"),
                        row[6],
                    ),
                }
                for row in cur.fetchall()
            ]


def update_physical_runtime_pins(session_id: str, pin_state: dict[str, bool]) -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pin_state FROM physical_runtime_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"error": f"Session {session_id} not found"}

            existing = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
            existing.update(pin_state)

            cur.execute(
                "UPDATE physical_runtime_sessions SET pin_state = %s WHERE id = %s",
                (json.dumps(existing), session_id),
            )
            conn.commit()
    return {"ok": True, "pin_state": existing}


def get_session_image_path(session_id: str, filename: str) -> Path | None:
    if ".." in filename or "/" in filename:
        return None
    path = _SESSION_IMAGES_DIR / f"phys_{session_id}" / filename
    return path if path.exists() else None


def _runtime_result_from_frame(
    selected_row: PhysicalEvalBoardRow,
    frame: VisualizedRuntimeFrame,
    *,
    elapsed_ms: float,
) -> dict[str, Any]:
    non_empty_square_count = sum(int(value != _EMPTY_CLASS_ID) for value in frame.gt_class_ids)
    temporal_non_empty_correct_count = _count_non_empty_correct(
        frame.temporal_class_ids,
        frame.gt_class_ids,
    )
    stateless_non_empty_correct_count = _count_non_empty_correct(
        frame.stateless_class_ids,
        frame.gt_class_ids,
    )

    return {
        "annotation_id": frame.annotation_id,
        "clip_path": selected_row.clip_path,
        "clip_filename": _clip_filename(selected_row),
        "frame_index": frame.frame_index,
        "board_path": frame.board_path,
        "source_video_id": selected_row.source_video_id,
        "gt_fen": _class_ids_to_fen(frame.gt_class_ids),
        "stateless_fen": _class_ids_to_fen(frame.stateless_class_ids),
        "temporal_fen": _class_ids_to_fen(frame.temporal_class_ids),
        "gt_change_count": frame.gt_change_count,
        "stateless_change_count": frame.stateless_change_count,
        "temporal_change_count": frame.temporal_change_count,
        "gt_changed_squares": _mask_to_square_names(frame.gt_changed_mask),
        "stateless_changed_squares": _mask_to_square_names(frame.stateless_changed_mask),
        "temporal_changed_squares": _mask_to_square_names(frame.temporal_changed_mask),
        "stateless_error_squares": _difference_square_names(
            frame.stateless_class_ids,
            frame.gt_class_ids,
        ),
        "temporal_error_squares": _difference_square_names(
            frame.temporal_class_ids,
            frame.gt_class_ids,
        ),
        "stateless_square_confidences": [
            round(float(confidence), 4) for confidence in frame.stateless_confidences
        ],
        "temporal_square_confidences": [
            round(float(confidence), 4) for confidence in frame.temporal_confidences
        ],
        "stateless_error_count": frame.stateless_error_count,
        "temporal_error_count": frame.temporal_error_count,
        "stateless_square_accuracy": round((64 - frame.stateless_error_count) / 64.0, 4),
        "temporal_square_accuracy": round((64 - frame.temporal_error_count) / 64.0, 4),
        "non_empty_square_count": non_empty_square_count,
        "stateless_non_empty_correct_count": stateless_non_empty_correct_count,
        "temporal_non_empty_correct_count": temporal_non_empty_correct_count,
        "stateless_non_empty_accuracy": round(
            stateless_non_empty_correct_count / non_empty_square_count,
            4,
        )
        if non_empty_square_count > 0
        else None,
        "temporal_non_empty_accuracy": round(
            temporal_non_empty_correct_count / non_empty_square_count,
            4,
        )
        if non_empty_square_count > 0
        else None,
        "stateless_exact_match": frame.stateless_error_count == 0,
        "temporal_exact_match": frame.temporal_error_count == 0,
        "stateless_mean_confidence": round(frame.stateless_mean_confidence, 4),
        "temporal_mean_confidence": round(frame.temporal_mean_confidence, 4),
        "elapsed_ms": elapsed_ms,
        "thumbnail_b64": _image_to_base64(Image.fromarray(frame.crop_rgb)),
    }



def _rows_by_clip() -> dict[str, list[PhysicalEvalBoardRow]]:
    return _group_rows_by_clip(PhysicalEvalBoardDataset().rows)



def _find_rows_by_annotation_ids(
    annotation_ids: list[str],
    rows_by_clip: dict[str, list[PhysicalEvalBoardRow]],
) -> dict[str, PhysicalEvalBoardRow]:
    rows_by_annotation_id = {
        row.annotation_id: row
        for clip_rows in rows_by_clip.values()
        for row in clip_rows
    }
    selected_rows: dict[str, PhysicalEvalBoardRow] = {}
    for annotation_id in annotation_ids:
        selected_row = rows_by_annotation_id.get(annotation_id)
        if selected_row is None:
            raise FileNotFoundError(f"Physical validation annotation {annotation_id} not found")
        if selected_row.frame_index is None:
            raise ValueError(f"Annotation {annotation_id} is missing frame_index")
        selected_rows[annotation_id] = selected_row
    return selected_rows



def _class_ids_to_fen(class_ids: tuple[int, ...]) -> str:
    ranks: list[str] = []
    for row in range(8):
        empty_run = 0
        rank_parts: list[str] = []
        for col in range(8):
            class_id = int(class_ids[row * 8 + col])
            class_name = SQUARE_CLASS_NAMES[class_id]
            if class_name == "empty":
                empty_run += 1
                continue
            if empty_run > 0:
                rank_parts.append(str(empty_run))
                empty_run = 0
            rank_parts.append(class_name)
        if empty_run > 0:
            rank_parts.append(str(empty_run))
        ranks.append("".join(rank_parts) or "8")
    return "/".join(ranks)


def _square_name(square_index: int) -> str:
    file = chr(ord("a") + (square_index % 8))
    rank = 8 - square_index // 8
    return f"{file}{rank}"


def _mask_to_square_names(mask: tuple[bool, ...]) -> list[str]:
    return [_square_name(index) for index, enabled in enumerate(mask) if enabled]


def _difference_square_names(
    predicted_class_ids: tuple[int, ...],
    target_class_ids: tuple[int, ...],
) -> list[str]:
    return [
        _square_name(index)
        for index, (predicted, target) in enumerate(zip(predicted_class_ids, target_class_ids))
        if predicted != target
    ]


def _count_non_empty_correct(
    predicted_class_ids: tuple[int, ...],
    target_class_ids: tuple[int, ...],
) -> int:
    return sum(
        int(predicted == target and target != _EMPTY_CLASS_ID)
        for predicted, target in zip(predicted_class_ids, target_class_ids)
    )


def _clip_filename(row: PhysicalEvalBoardRow) -> str:
    if row.clip_path:
        return Path(row.clip_path).name
    return Path(row.board_path).name


def _default_runtime_version() -> str | None:
    metadata_path = _WEIGHTS_DIR / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None
    version = metadata.get("version")
    return str(version) if isinstance(version, str) else None


def _runtime_model_label(path: Path, *, default_version: str | None) -> str:
    relative_path = _relative_to_project(path)
    if path == _WEIGHTS_DIR / "best.pt":
        return f"default · {default_version}" if default_version else "default"
    if path.is_relative_to(_WEIGHTS_DIR):
        return path.stem
    if path.name in {"board_probe.pt", "linear_probe.pt"}:
        return path.parent.name
    return Path(relative_path).stem or path.name


def _relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _isoformat_mtime(path: Path) -> str | None:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))
    except OSError:
        return None


def _evaluation_model_path(
    evaluation_per_class: dict[str, Any],
    notes: Any,
) -> str | None:
    raw_model_path = evaluation_per_class.get("model_path")
    if isinstance(raw_model_path, str) and raw_model_path.strip():
        return raw_model_path.strip()
    if isinstance(notes, str):
        return _resolve_legacy_model_path(notes.strip())
    return None


def _evaluation_model_label(
    evaluation_per_class: dict[str, Any],
    notes: Any,
    model_path: str | None,
) -> str | None:
    raw_model_label = evaluation_per_class.get("model_label")
    if isinstance(raw_model_label, str) and raw_model_label.strip():
        return raw_model_label.strip()
    if isinstance(notes, str) and notes.strip():
        return notes.strip()
    if model_path is not None:
        return _runtime_model_label(Path(PROJECT_ROOT / model_path), default_version=_default_runtime_version())
    return None


def _resolve_legacy_model_path(note: str) -> str | None:
    if not note:
        return None

    candidate = Path(note)
    if candidate.is_absolute() and candidate.exists():
        return _relative_to_project(candidate)

    relative_candidate = PROJECT_ROOT / note
    if relative_candidate.exists():
        return _relative_to_project(relative_candidate)

    version_candidate = _WEIGHTS_DIR / f"{note}.pt"
    if version_candidate.exists():
        return _relative_to_project(version_candidate)

    if note == _default_runtime_version() and (_WEIGHTS_DIR / "best.pt").exists():
        return _relative_to_project(_WEIGHTS_DIR / "best.pt")

    return None


def _save_session_image(session_id: str, name: str, b64_data: str) -> str:
    out_dir = _SESSION_IMAGES_DIR / f"phys_{session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.png"
    (out_dir / filename).write_bytes(base64.b64decode(b64_data))
    return filename


def _safe_filename_stem(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_") or "frame"


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
