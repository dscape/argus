from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pipeline.physical.piece_projection import (
    DEFAULT_PIECE_HEIGHT,
    camera_pose_from_corners,
    default_camera_matrix,
    piece_bbox_from_projection,
    project_piece_box,
)
from pipeline.physical.shared.real_board_data import (
    infer_channel_corner_templates,
    replay_clip_display_fens,
)
from pipeline.shared import SQUARE_CLASS_NAMES, fen_to_square_labels
from study.templates.inference.embedder import (
    DEFAULT_ENCODER_TYPE,
    DEFAULT_INPUT_SIZE,
    get_embedder,
)
from study.templates.shared import PROJECT_ROOT, load_base_head_data_module

STANDARD_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
PIECE_TYPES = tuple(SQUARE_CLASS_NAMES[1:])
STARTING_POSITION_PIECE_COUNTS = {
    "P": 8,
    "N": 2,
    "B": 2,
    "R": 2,
    "Q": 1,
    "K": 1,
    "p": 8,
    "n": 2,
    "b": 2,
    "r": 2,
    "q": 1,
    "k": 1,
}
_ANNOTATION_ROOTS = (
    PROJECT_ROOT / "data" / "physical" / "train",
    PROJECT_ROOT / "data" / "physical" / "val",
)


@dataclass(frozen=True)
class TemplateBankConfig:
    encoder_type: str = DEFAULT_ENCODER_TYPE
    model_name: str | None = None
    input_size: int = DEFAULT_INPUT_SIZE
    device: str = "cpu"
    piece_height: float = DEFAULT_PIECE_HEIGHT
    flip_left_half: bool = True
    jitter_variations: int = 9
    jitter_pixels: int = 4
    batch_size: int = 32
    preview_samples_per_piece_type: int = 4
    max_base_crops_per_piece_type: int | None = 4
    native_frame_stride: int = 3
    temporal_consistency_weight: float = 8.0


@dataclass(frozen=True)
class ReplayClipSelection:
    clip_path: str
    tournament_id: str
    source_video_id: str | None
    source_channel_handle: str
    corners: tuple[tuple[float, float], ...]
    rows: list[Any]
    source_frame_indices: list[int]


@dataclass(frozen=True)
class TemplateCropCandidate:
    piece_type: str
    square_index: int
    square_name: str
    row_id: str
    source_frame_index: int | None
    crop_bgr: np.ndarray
    overlay_bgr: np.ndarray
    score: float


def build_template_bank(
    rows: list[Any],
    *,
    tournament_id: str,
    output_path: str | Path,
    preview_path: str | Path | None = None,
    config: TemplateBankConfig | None = None,
    embed_fn: Callable[[np.ndarray], torch.Tensor] | None = None,
) -> dict[str, Any]:
    resolved_config = config or TemplateBankConfig()
    _validate_builder_config(resolved_config)
    if not rows:
        raise ValueError("rows must be non-empty")

    candidates_by_piece_type = _collect_candidates(rows, config=resolved_config)
    scored_candidates_by_piece_type = _apply_temporal_consistency_scores(
        candidates_by_piece_type,
        weight=resolved_config.temporal_consistency_weight,
    )
    selected_by_piece_type = {
        piece_type: _select_candidates(
            scored_candidates_by_piece_type[piece_type],
            max_count=resolved_config.max_base_crops_per_piece_type,
        )
        for piece_type in PIECE_TYPES
    }

    resolved_model_name = resolved_config.model_name
    if embed_fn is None:
        resolved_model_name = get_embedder(
            encoder_type=resolved_config.encoder_type,
            model_name=resolved_config.model_name,
            input_size=resolved_config.input_size,
            device=resolved_config.device,
        ).model_name

    embeddings_by_piece_type: dict[str, torch.Tensor] = {}
    metadata_by_piece_type: dict[str, list[dict[str, Any]]] = {}
    for piece_type in PIECE_TYPES:
        selected_candidates = selected_by_piece_type[piece_type]
        jittered_crops: list[np.ndarray] = []
        template_metadata: list[dict[str, Any]] = []
        for candidate in selected_candidates:
            for jitter_index, (jitter_dx, jitter_dy, jitter_crop) in enumerate(
                _generate_jitter_variants(
                    candidate.crop_bgr,
                    num_variations=resolved_config.jitter_variations,
                    max_offset_pixels=resolved_config.jitter_pixels,
                )
            ):
                jittered_crops.append(jitter_crop)
                template_metadata.append(
                    {
                        "row_id": candidate.row_id,
                        "square": candidate.square_name,
                        "source_frame_index": candidate.source_frame_index,
                        "score": candidate.score,
                        "jitter_index": jitter_index,
                        "jitter_dx": jitter_dx,
                        "jitter_dy": jitter_dy,
                    }
                )
        embeddings_by_piece_type[piece_type] = _embed_piece_crops(
            jittered_crops,
            config=resolved_config,
            embed_fn=embed_fn,
        )
        metadata_by_piece_type[piece_type] = template_metadata

    embedding_dim = next(
        (int(tensor.shape[1]) for tensor in embeddings_by_piece_type.values() if tensor.ndim == 2),
        0,
    )
    payload = {
        "tournament_id": tournament_id,
        "encoder_config": {
            "encoder_type": resolved_config.encoder_type,
            "model_name": resolved_model_name,
            "input_size": resolved_config.input_size,
            "device": resolved_config.device,
            "embedding_dim": embedding_dim,
        },
        "builder_config": {
            "piece_height": resolved_config.piece_height,
            "flip_left_half": resolved_config.flip_left_half,
            "jitter_variations": resolved_config.jitter_variations,
            "jitter_pixels": resolved_config.jitter_pixels,
            "batch_size": resolved_config.batch_size,
            "frame_count": len(rows),
            "max_base_crops_per_piece_type": resolved_config.max_base_crops_per_piece_type,
            "native_frame_stride": resolved_config.native_frame_stride,
            "temporal_consistency_weight": resolved_config.temporal_consistency_weight,
        },
        "frame_ids": [str(row.row_id) for row in rows],
        "row_lookup": {
            str(row.row_id): _row_to_payload(row) for row in rows
        },
        "embeddings_by_piece_type": embeddings_by_piece_type,
        "candidate_base_crop_counts_by_piece_type": {
            piece_type: len(scored_candidates_by_piece_type[piece_type])
            for piece_type in PIECE_TYPES
        },
        "base_crop_counts_by_piece_type": {
            piece_type: len(selected_by_piece_type[piece_type]) for piece_type in PIECE_TYPES
        },
        "embedding_counts_by_piece_type": {
            piece_type: int(embeddings_by_piece_type[piece_type].shape[0])
            for piece_type in PIECE_TYPES
        },
        "template_metadata_by_piece_type": metadata_by_piece_type,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    if preview_path is not None:
        preview_image = render_template_preview(
            tournament_id=tournament_id,
            selected_by_piece_type=selected_by_piece_type,
            candidate_counts_by_piece_type={
                piece_type: len(scored_candidates_by_piece_type[piece_type])
                for piece_type in PIECE_TYPES
            },
            frame_count=len(rows),
            config=resolved_config,
        )
        preview_output = Path(preview_path)
        preview_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_output), preview_image):
            raise ValueError(f"Failed to write preview image: {preview_output}")

    return payload


def build_replay_clip_selection(
    clip_path: str | Path,
    *,
    tournament_id: str | None = None,
    corners: tuple[tuple[float, float], ...] | None = None,
    max_frames: int | None = None,
    native_frame_stride: int = 3,
) -> ReplayClipSelection:
    base_head_data = load_base_head_data_module()
    resolved_clip_path = Path(clip_path)
    if not resolved_clip_path.is_absolute():
        resolved_clip_path = (PROJECT_ROOT / resolved_clip_path).resolve()
    clip = torch.load(resolved_clip_path, map_location="cpu", weights_only=False)
    if not isinstance(clip, dict):
        raise ValueError(f"Invalid clip payload: {resolved_clip_path}")

    initial_board_fen = clip.get("initial_board_fen")
    if _piece_placement_fen(initial_board_fen) != STANDARD_START_FEN:
        raise ValueError(
            "Replay clip does not start from the standard initial position: "
            f"{resolved_clip_path.name}"
        )

    source_channel_handle = clip.get("source_channel_handle")
    if not isinstance(source_channel_handle, str) or not source_channel_handle:
        raise ValueError(f"Replay clip is missing source_channel_handle: {resolved_clip_path}")

    relative_clip_path = _path_relative_to_project_root(resolved_clip_path)
    selection_tournament_id = tournament_id or str(
        clip.get("source_video_id") or source_channel_handle
    )
    starting_labels = _flatten_fen_labels(STANDARD_START_FEN)

    native_rows = _expand_to_native_start_interval_rows(
        base_head_data,
        clip=clip,
        clip_path=relative_clip_path,
        labels=starting_labels,
        max_frames=max_frames,
        native_frame_stride=native_frame_stride,
    )
    if native_rows is not None:
        anchor = native_rows[0]
        return ReplayClipSelection(
            clip_path=relative_clip_path,
            tournament_id=selection_tournament_id,
            source_video_id=(
                None if clip.get("source_video_id") is None else str(clip.get("source_video_id"))
            ),
            source_channel_handle=source_channel_handle,
            corners=_full_frame_corners(anchor),
            rows=native_rows,
            source_frame_indices=[
                int(row.source_frame_index)
                for row in native_rows
                if row.source_frame_index is not None
            ],
        )

    resolved_corners = corners or _resolve_clip_corners(source_channel_handle)
    frame_fens = replay_clip_display_fens(clip)
    sampled_frame_indices = _sampled_frame_indices(clip)
    rows: list[Any] = []
    source_frame_indices: list[int] = []

    for sample_index, fen in enumerate(frame_fens):
        if _piece_placement_fen(fen) != STANDARD_START_FEN:
            continue
        source_frame_index = sampled_frame_indices[sample_index]
        rows.append(
            base_head_data.BoardRow(
                row_id=f"{resolved_clip_path.name}:{sample_index}",
                clip_path=relative_clip_path,
                frame_index=sample_index,
                image_path=None,
                source_video_id=(
                    None
                    if clip.get("source_video_id") is None
                    else str(clip.get("source_video_id"))
                ),
                source_frame_index=source_frame_index,
                corners=resolved_corners,
                labels=starting_labels,
            )
        )
        source_frame_indices.append(int(source_frame_index))
        if max_frames is not None and len(rows) >= max_frames:
            break

    if not rows:
        raise ValueError(
            "No sampled frames remained in the standard starting position: "
            f"{clip_path}"
        )

    return ReplayClipSelection(
        clip_path=relative_clip_path,
        tournament_id=selection_tournament_id,
        source_video_id=(
            None if clip.get("source_video_id") is None else str(clip.get("source_video_id"))
        ),
        source_channel_handle=source_channel_handle,
        corners=resolved_corners,
        rows=rows,
        source_frame_indices=source_frame_indices,
    )


def build_source_video_selection(
    source_video_id: str,
    *,
    source_frame_indices: list[int],
    tournament_id: str | None = None,
) -> ReplayClipSelection:
    if not source_frame_indices:
        raise ValueError("source_frame_indices must be non-empty")

    base_head_data = load_base_head_data_module()
    anchor = _load_anchor_annotation_row_for_source_video(
        base_head_data,
        source_video_id=source_video_id,
    )
    starting_labels = _flatten_fen_labels(STANDARD_START_FEN)
    rows: list[Any] = []
    for source_frame_index in source_frame_indices:
        rows.append(
            base_head_data.BoardRow(
                row_id=f"{source_video_id}:native:{source_frame_index}",
                clip_path=anchor.clip_path,
                frame_index=anchor.frame_index,
                image_path=None,
                source_video_id=source_video_id,
                source_frame_index=int(source_frame_index),
                corners=anchor.corners,
                labels=starting_labels,
                native_corners=anchor.native_corners,
                native_image_bbox=anchor.native_image_bbox,
            )
        )

    return ReplayClipSelection(
        clip_path=str(anchor.clip_path),
        tournament_id=tournament_id or source_video_id,
        source_video_id=source_video_id,
        source_channel_handle=source_video_id,
        corners=_full_frame_corners(anchor),
        rows=rows,
        source_frame_indices=[int(index) for index in source_frame_indices],
    )


def rebuild_template_crop(
    template_bank: dict[str, Any],
    metadata: dict[str, Any],
    *,
    output_size: int | None = None,
) -> np.ndarray:
    row_lookup = template_bank.get("row_lookup")
    if not isinstance(row_lookup, dict):
        raise ValueError("Template bank is missing row_lookup")
    row_id = metadata.get("row_id")
    if not isinstance(row_id, str) or row_id not in row_lookup:
        raise ValueError(f"Template bank metadata has unknown row_id: {row_id!r}")

    base_head_data = load_base_head_data_module()
    row = _row_from_payload(base_head_data, row_lookup[row_id])
    native_loader = base_head_data.NativeFrameLoader()
    try:
        frame_bgr, corners = base_head_data.load_row_frame_and_corners(
            row,
            clip_cache={},
            native_loader=native_loader,
        )
    finally:
        native_loader.close()

    square_name = metadata.get("square")
    if not isinstance(square_name, str):
        raise ValueError(f"Template bank metadata is missing square: {metadata!r}")
    square_index = base_head_data.square_name_to_index(square_name)
    builder_config = dict(template_bank.get("builder_config", {}))
    encoder_config = dict(template_bank.get("encoder_config", {}))
    crop = base_head_data.extract_study_piece_crop(
        frame_bgr,
        corners,
        row=square_index // 8,
        col=square_index % 8,
        output_size=int(output_size or encoder_config.get("input_size", 224)),
        piece_height=float(builder_config.get("piece_height", DEFAULT_PIECE_HEIGHT)),
        flip_left_half=bool(builder_config.get("flip_left_half", True)),
    )
    jitter_dx = int(metadata.get("jitter_dx", 0))
    jitter_dy = int(metadata.get("jitter_dy", 0))
    if jitter_dx != 0 or jitter_dy != 0:
        crop = _translate_crop(crop, dx=jitter_dx, dy=jitter_dy)
    return crop


def render_template_preview(
    *,
    tournament_id: str,
    selected_by_piece_type: dict[str, list[TemplateCropCandidate]],
    candidate_counts_by_piece_type: dict[str, int],
    frame_count: int,
    config: TemplateBankConfig,
) -> np.ndarray:
    crop_size = config.input_size
    title_height = 56
    header_width = 220
    max_columns = max(
        1,
        min(
            config.preview_samples_per_piece_type,
            max((len(selected_by_piece_type[piece_type]) for piece_type in PIECE_TYPES), default=1),
        ),
    )
    row_height = crop_size + 16
    width = header_width + (max_columns * crop_size)
    height = title_height + (len(PIECE_TYPES) * row_height)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)

    cv2.putText(
        canvas,
        f"selected template preview | {tournament_id} | frames={frame_count}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            "each tile: square @ source-frame | projected cuboid outline | "
            "score-ranked selection"
        ),
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    for row_index, piece_type in enumerate(PIECE_TYPES):
        top = title_height + (row_index * row_height)
        center_y = top + crop_size // 2
        selected_count = len(selected_by_piece_type.get(piece_type, []))
        candidate_count = int(candidate_counts_by_piece_type.get(piece_type, 0))
        cv2.putText(
            canvas,
            f"{piece_type} ({selected_count}/{candidate_count})",
            (12, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        for column_index, candidate in enumerate(
            selected_by_piece_type.get(piece_type, [])[:max_columns]
        ):
            left = header_width + (column_index * crop_size)
            canvas[top : top + crop_size, left : left + crop_size] = candidate.overlay_bgr
            label = candidate.square_name
            if candidate.source_frame_index is not None:
                label = f"{label} @{candidate.source_frame_index}"
            cv2.putText(
                canvas,
                label,
                (left + 8, top + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"score={candidate.score:.3f}",
                (left + 8, top + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return canvas


def _collect_candidates(
    rows: list[Any],
    *,
    config: TemplateBankConfig,
) -> dict[str, list[TemplateCropCandidate]]:
    base_head_data = load_base_head_data_module()
    clip_cache: dict[Path, dict[str, Any]] = {}
    native_loader = base_head_data.NativeFrameLoader()
    candidates_by_piece_type: dict[str, list[TemplateCropCandidate]] = defaultdict(list)

    try:
        for row in rows:
            frame_bgr, corners = base_head_data.load_row_frame_and_corners(
                row,
                clip_cache=clip_cache,
                native_loader=native_loader,
            )
            pose = camera_pose_from_corners(corners, K=default_camera_matrix(frame_bgr.shape))
            for square_index, label in enumerate(row.labels):
                if int(label) <= 0:
                    continue
                piece_type = SQUARE_CLASS_NAMES[int(label)]
                candidate = _build_candidate(
                    base_head_data,
                    row=row,
                    frame_bgr=frame_bgr,
                    corners=corners,
                    pose=pose,
                    square_index=square_index,
                    piece_type=piece_type,
                    config=config,
                )
                candidates_by_piece_type[piece_type].append(candidate)
    finally:
        native_loader.close()

    return {piece_type: candidates_by_piece_type[piece_type] for piece_type in PIECE_TYPES}


def _apply_temporal_consistency_scores(
    candidates_by_piece_type: dict[str, list[TemplateCropCandidate]],
    *,
    weight: float,
) -> dict[str, list[TemplateCropCandidate]]:
    if weight <= 0.0:
        return candidates_by_piece_type

    scored_by_piece_type: dict[str, list[TemplateCropCandidate]] = {}
    for piece_type, candidates in candidates_by_piece_type.items():
        grouped_by_square: dict[int, list[TemplateCropCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped_by_square[candidate.square_index].append(candidate)

        updated_candidates: list[TemplateCropCandidate] = []
        for square_candidates in grouped_by_square.values():
            gray_stack = np.stack(
                [
                    cv2.cvtColor(candidate.crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    / 255.0
                    for candidate in square_candidates
                ],
                axis=0,
            )
            median_gray = np.median(gray_stack, axis=0)
            for candidate, gray in zip(square_candidates, gray_stack, strict=False):
                deviation = float(np.abs(gray - median_gray).mean())
                adjusted_score = candidate.score / (1.0 + (weight * deviation))
                updated_candidates.append(replace(candidate, score=adjusted_score))
        scored_by_piece_type[piece_type] = updated_candidates
    return scored_by_piece_type


def _build_candidate(
    base_head_data: Any,
    *,
    row: Any,
    frame_bgr: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    pose: Any,
    square_index: int,
    piece_type: str,
    config: TemplateBankConfig,
) -> TemplateCropCandidate:
    row_index = square_index // 8
    col_index = square_index % 8
    crop_bgr = base_head_data.extract_study_piece_crop(
        frame_bgr,
        corners,
        row=row_index,
        col=col_index,
        output_size=config.input_size,
        piece_height=config.piece_height,
        flip_left_half=config.flip_left_half,
    )
    projected_box = project_piece_box(
        pose,
        row=row_index,
        col=col_index,
        piece_height=config.piece_height,
        corners=corners,
    )
    bbox = piece_bbox_from_projection(projected_box)
    flip = bool(config.flip_left_half and col_index < 4)
    hull = _project_hull_to_crop_canvas(
        projected_box,
        bbox=bbox,
        frame_shape=frame_bgr.shape,
        output_size=config.input_size,
        flip_horizontally=flip,
    )
    score = _candidate_quality_score(crop_bgr, hull)
    overlay_bgr = _render_candidate_overlay(crop_bgr, hull)
    return TemplateCropCandidate(
        piece_type=piece_type,
        square_index=square_index,
        square_name=base_head_data.index_to_square_name(square_index),
        row_id=str(row.row_id),
        source_frame_index=(
            None
            if getattr(row, "source_frame_index", None) is None
            else int(row.source_frame_index)
        ),
        crop_bgr=crop_bgr,
        overlay_bgr=overlay_bgr,
        score=score,
    )


def _select_candidates(
    candidates: list[TemplateCropCandidate],
    *,
    max_count: int | None,
) -> list[TemplateCropCandidate]:
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    if max_count is None:
        return ranked
    return ranked[:max_count]


def _candidate_quality_score(crop_bgr: np.ndarray, hull: np.ndarray) -> float:
    mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    if int(mask.sum()) == 0:
        return 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    score = float(magnitude[mask > 0].mean())

    touches_border = (
        float(hull[:, 0].min()) <= 1.0
        or float(hull[:, 1].min()) <= 1.0
        or float(hull[:, 0].max()) >= crop_bgr.shape[1] - 2.0
        or float(hull[:, 1].max()) >= crop_bgr.shape[0] - 2.0
    )
    if touches_border:
        score *= 0.85
    return score


def _render_candidate_overlay(crop_bgr: np.ndarray, hull: np.ndarray) -> np.ndarray:
    overlay = crop_bgr.copy()
    fill = overlay.copy()
    cv2.fillConvexPoly(fill, np.round(hull).astype(np.int32), (0, 96, 0))
    overlay = cv2.addWeighted(fill, 0.18, overlay, 0.82, 0.0)
    cv2.polylines(
        overlay,
        [np.round(hull).astype(np.int32)],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
    )
    return overlay


def _project_hull_to_crop_canvas(
    projected_box: np.ndarray,
    *,
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
    output_size: int,
    flip_horizontally: bool,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    height, width = frame_shape[:2]
    clipped_xmin = max(0, int(np.floor(xmin)))
    clipped_ymin = max(0, int(np.floor(ymin)))
    clipped_xmax = min(width, int(np.ceil(xmax)))
    clipped_ymax = min(height, int(np.ceil(ymax)))
    crop_width = max(1, clipped_xmax - clipped_xmin)
    crop_height = max(1, clipped_ymax - clipped_ymin)
    scale = float(output_size) / float(max(crop_height, crop_width))
    new_height = max(1, int(round(crop_height * scale)))
    new_width = max(1, int(round(crop_width * scale)))
    y_offset = output_size - new_height

    local_points = projected_box.astype(np.float32).copy()
    local_points[:, 0] -= float(clipped_xmin)
    local_points[:, 1] -= float(clipped_ymin)
    local_points[:, 0] = np.clip(local_points[:, 0], 0.0, float(crop_width - 1))
    local_points[:, 1] = np.clip(local_points[:, 1], 0.0, float(crop_height - 1))
    local_points *= scale
    if flip_horizontally:
        local_points[:, 0] = float(new_width - 1) - local_points[:, 0]
    local_points[:, 1] += float(y_offset)
    hull = cv2.convexHull(local_points.reshape(-1, 1, 2)).reshape(-1, 2)
    return hull.astype(np.float32)


def _embed_piece_crops(
    crops: list[np.ndarray],
    *,
    config: TemplateBankConfig,
    embed_fn: Callable[[np.ndarray], torch.Tensor] | None,
) -> torch.Tensor:
    if embed_fn is not None:
        if not crops:
            return torch.empty((0, 0), dtype=torch.float32)
        embeddings = [embed_fn(crop).detach().cpu().flatten() for crop in crops]
        return torch.stack(embeddings, dim=0)

    embedder = get_embedder(
        encoder_type=config.encoder_type,
        model_name=config.model_name,
        input_size=config.input_size,
        device=config.device,
    )
    batches: list[torch.Tensor] = []
    for start in range(0, len(crops), config.batch_size):
        batches.append(embedder.embed_many(crops[start : start + config.batch_size]))
    if not batches:
        return torch.empty((0, embedder.embedding_dim), dtype=torch.float32)
    return torch.cat(batches, dim=0)


def _generate_jitter_variants(
    crop: np.ndarray,
    *,
    num_variations: int,
    max_offset_pixels: int,
) -> list[tuple[int, int, np.ndarray]]:
    offsets = _jitter_offsets(num_variations=num_variations, max_offset_pixels=max_offset_pixels)
    return [(dx, dy, _translate_crop(crop, dx=dx, dy=dy)) for dx, dy in offsets]


def _jitter_offsets(*, num_variations: int, max_offset_pixels: int) -> list[tuple[int, int]]:
    if num_variations <= 0:
        raise ValueError(f"num_variations must be > 0, got {num_variations}")
    if num_variations == 1 or max_offset_pixels <= 0:
        return [(0, 0)] * num_variations

    side = max(1, math.ceil(math.sqrt(num_variations)))
    axis = np.linspace(-max_offset_pixels, max_offset_pixels, num=side)
    offsets: list[tuple[int, int]] = []
    for dy in axis:
        for dx in axis:
            offsets.append((int(round(dx)), int(round(dy))))
    center_first = sorted(
        offsets,
        key=lambda item: (
            abs(item[0]) + abs(item[1]),
            abs(item[1]),
            abs(item[0]),
        ),
    )
    return center_first[:num_variations]


def _translate_crop(crop: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    return cv2.warpAffine(
        crop,
        matrix,
        (crop.shape[1], crop.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _expand_to_native_start_interval_rows(
    base_head_data: Any,
    *,
    clip: dict[str, Any],
    clip_path: str,
    labels: tuple[int, ...],
    max_frames: int | None,
    native_frame_stride: int,
) -> list[Any] | None:
    anchor_rows = _load_start_annotation_rows_for_clip(
        base_head_data,
        clip_path=clip_path,
        labels=labels,
    )
    if not anchor_rows:
        return None

    anchor = anchor_rows[0]
    if anchor.source_frame_index is None or anchor.source_video_id is None:
        return None

    move_frame_indices = _int_list(clip.get("move_frame_indices"))
    first_move_frame = min(move_frame_indices) if move_frame_indices else None
    start_frame = int(anchor.source_frame_index)
    end_frame = first_move_frame if first_move_frame is not None else start_frame + 1
    if end_frame <= start_frame:
        return None

    rows: list[Any] = []
    for source_frame_index in range(start_frame, end_frame, native_frame_stride):
        rows.append(
            base_head_data.BoardRow(
                row_id=f"{Path(clip_path).name}:native:{source_frame_index}",
                clip_path=clip_path,
                frame_index=anchor.frame_index,
                image_path=None,
                source_video_id=str(anchor.source_video_id),
                source_frame_index=source_frame_index,
                corners=anchor.corners,
                labels=labels,
                native_corners=anchor.native_corners,
                native_image_bbox=anchor.native_image_bbox,
            )
        )
        if max_frames is not None and len(rows) >= max_frames:
            break
    return rows or None


def _load_start_annotation_rows_for_clip(
    base_head_data: Any,
    *,
    clip_path: str,
    labels: tuple[int, ...],
) -> list[Any]:
    matches: list[Any] = []
    for annotation_root in _ANNOTATION_ROOTS:
        if not annotation_root.exists():
            continue
        for row in base_head_data.load_annotation_rows(annotation_root):
            if row.clip_path != clip_path or row.labels != labels:
                continue
            if row.native_corners is None or row.native_image_bbox is None:
                continue
            matches.append(row)
    matches.sort(
        key=lambda row: (
            -1 if row.source_frame_index is None else int(row.source_frame_index),
            str(row.row_id),
        )
    )
    return matches


def _full_frame_corners(row: Any) -> tuple[tuple[float, float], ...]:
    if row.native_corners is None or row.native_image_bbox is None:
        return row.corners
    x_off, y_off, _width, _height = row.native_image_bbox
    return tuple((float(x + x_off), float(y + y_off)) for x, y in row.native_corners)


def _row_to_payload(row: Any) -> dict[str, Any]:
    return {
        "row_id": str(row.row_id),
        "clip_path": row.clip_path,
        "frame_index": row.frame_index,
        "image_path": row.image_path,
        "source_video_id": row.source_video_id,
        "source_frame_index": row.source_frame_index,
        "corners": row.corners,
        "labels": row.labels,
        "native_corners": row.native_corners,
        "native_image_bbox": row.native_image_bbox,
    }


def _row_from_payload(base_head_data: Any, payload: dict[str, Any]) -> Any:
    return base_head_data.BoardRow(
        row_id=str(payload["row_id"]),
        clip_path=payload.get("clip_path"),
        frame_index=(
            None if payload.get("frame_index") is None else int(payload["frame_index"])
        ),
        image_path=payload.get("image_path"),
        source_video_id=(
            None
            if payload.get("source_video_id") is None
            else str(payload["source_video_id"])
        ),
        source_frame_index=(
            None
            if payload.get("source_frame_index") is None
            else int(payload["source_frame_index"])
        ),
        corners=tuple((float(x), float(y)) for x, y in payload["corners"]),
        labels=tuple(int(value) for value in payload["labels"]),
        native_corners=(
            None
            if payload.get("native_corners") is None
            else tuple((float(x), float(y)) for x, y in payload["native_corners"])
        ),
        native_image_bbox=(
            None
            if payload.get("native_image_bbox") is None
            else tuple(int(value) for value in payload["native_image_bbox"])
        ),
    )


def _load_anchor_annotation_row_for_source_video(
    base_head_data: Any,
    *,
    source_video_id: str,
) -> Any:
    matches: list[Any] = []
    for annotation_root in _ANNOTATION_ROOTS:
        if not annotation_root.exists():
            continue
        for row in base_head_data.load_annotation_rows(annotation_root):
            if row.source_video_id != source_video_id:
                continue
            if row.native_corners is None or row.native_image_bbox is None:
                continue
            matches.append(row)
    if not matches:
        raise ValueError(
            f"No annotation row with native metadata found for source_video_id={source_video_id!r}"
        )
    matches.sort(
        key=lambda row: (
            -1 if row.source_frame_index is None else int(row.source_frame_index),
            str(row.row_id),
        )
    )
    return matches[0]


def _resolve_clip_corners(source_channel_handle: str) -> tuple[tuple[float, float], ...]:
    corner_templates = infer_channel_corner_templates()
    corners = corner_templates.get(source_channel_handle)
    if corners is None:
        raise ValueError(
            "No existing board detector output found for channel handle "
            f"{source_channel_handle!r}"
        )
    return corners


def _flatten_fen_labels(fen: str) -> tuple[int, ...]:
    return tuple(int(label) for row in fen_to_square_labels(fen) for label in row)


def _piece_placement_fen(fen: str | None) -> str | None:
    if fen is None:
        return None
    return str(fen).split(" ", 1)[0]


def _path_relative_to_project_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _sampled_frame_indices(clip: dict[str, Any]) -> list[int]:
    frames = clip.get("frames")
    if not isinstance(frames, torch.Tensor):
        return []
    frame_indices = clip.get("frame_indices")
    if isinstance(frame_indices, torch.Tensor):
        values = [int(value) for value in frame_indices.tolist()]
        if len(values) == int(frames.shape[0]):
            return values
    if isinstance(frame_indices, list) and len(frame_indices) == int(frames.shape[0]):
        return [int(value) for value in frame_indices]
    return list(range(int(frames.shape[0])))


def _int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.tolist()]
    if isinstance(value, list):
        return [int(item) for item in value]
    return []


def _validate_builder_config(config: TemplateBankConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
    if config.preview_samples_per_piece_type <= 0:
        raise ValueError(
            "preview_samples_per_piece_type must be > 0, got "
            f"{config.preview_samples_per_piece_type}"
        )
    if (
        config.max_base_crops_per_piece_type is not None
        and config.max_base_crops_per_piece_type <= 0
    ):
        raise ValueError(
            "max_base_crops_per_piece_type must be > 0 when set, got "
            f"{config.max_base_crops_per_piece_type}"
        )
    if config.native_frame_stride <= 0:
        raise ValueError(f"native_frame_stride must be > 0, got {config.native_frame_stride}")
    if config.temporal_consistency_weight < 0.0:
        raise ValueError(
            "temporal_consistency_weight must be >= 0, got "
            f"{config.temporal_consistency_weight}"
        )


__all__ = [
    "PIECE_TYPES",
    "STANDARD_START_FEN",
    "STARTING_POSITION_PIECE_COUNTS",
    "ReplayClipSelection",
    "TemplateBankConfig",
    "build_replay_clip_selection",
    "build_source_video_selection",
    "build_template_bank",
    "rebuild_template_crop",
    "render_template_preview",
]
