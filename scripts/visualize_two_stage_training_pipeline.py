#!/usr/bin/env python3
"""Generate a visual walkthrough of the two-stage physical training pipeline.

This focuses on the crop-based geometry path used by:
- `pipeline/physical/two_stage/classifier_data.py`
- `pipeline/physical/two_stage/reader.py`

It shows, for one annotated board frame:
1. which source frame is actually used for training,
2. how occupancy and piece bboxes are computed,
3. the resulting occupancy and piece crops,
4. how labels are mapped into the two classifier tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.piece_projection import (
    DEFAULT_PIECE_HEIGHT,
    camera_pose_from_corners,
    default_camera_matrix,
    piece_bbox_from_projection,
    project_piece_box,
    project_square_base_quad,
    square_bbox_from_corners,
)
from pipeline.physical.shared.annotation_rows import (
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.two_stage.classifier_data import (
    DEFAULT_OCCUPANCY_CROP_SIZE,
    DEFAULT_PIECE_CROP_SIZE,
    NativeFrameLoader,
)
from pipeline.physical.two_stage.classifiers import (
    square_class_to_occupancy_label,
    square_class_to_piece_label,
)
from pipeline.shared import SQUARE_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / ".agents" / "memory" / "attachments"
_OCCUPANCY_COLOR = (70, 200, 255)
_PIECE_COLOR = (255, 170, 60)
_BOARD_COLOR = (80, 220, 100)
_CORNER_COLOR = (255, 255, 255)
_SELECTED_COLORS = {
    "c4": (255, 90, 90),
    "b4": (90, 255, 120),
    "g5": (255, 220, 90),
}


@dataclass(frozen=True)
class SelectedSquare:
    square_name: str
    square_index: int
    label_name: str
    occupancy_label_name: str
    piece_label_name: str | None
    flip_piece_crop: bool
    occupancy_bbox: tuple[float, float, float, float]
    piece_bbox: tuple[float, float, float, float] | None
    occupancy_raw_crop_rgb: np.ndarray
    occupancy_canvas_rgb: np.ndarray
    piece_raw_crop_rgb: np.ndarray | None
    piece_canvas_rgb: np.ndarray | None


@dataclass(frozen=True)
class PipelineContext:
    annotation_id: str
    split_name: str
    row: Any
    source_kind: str
    source_frame_bgr: np.ndarray
    clip_frame_bgr: np.ndarray
    training_corners: tuple[tuple[float, float], ...]
    clip_corners: tuple[tuple[float, float], ...]
    occupancy_bboxes: tuple[tuple[float, float, float, float], ...]
    piece_bboxes: tuple[tuple[float, float, float, float], ...]
    occupancy_crops_rgb: tuple[np.ndarray, ...]
    piece_crops_rgb_by_square: dict[int, np.ndarray]
    occupied_square_indices: tuple[int, ...]
    selected_squares: tuple[SelectedSquare, ...]


@dataclass(frozen=True)
class CropDebug:
    raw_crop_rgb: np.ndarray
    canvas_rgb: np.ndarray
    clipped_bbox: tuple[int, int, int, int]
    resized_size: tuple[int, int]
    scale: float


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_context(
        annotation_id=args.annotation_id,
        train_root=args.physical_train_root,
        val_root=args.physical_val_root,
        selected_square_names=[name.strip() for name in args.selected_squares.split(",") if name],
    )

    write_outputs(context, output_dir=output_dir)
    print(output_dir / "summary.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize the crop-based two-stage physical training pipeline for one frame."
    )
    parser.add_argument(
        "--annotation-id",
        type=str,
        default="clip_overlay_cQAedm_gWrw_clip67_2_frame0000",
    )
    parser.add_argument(
        "--physical-train-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "physical" / "train",
    )
    parser.add_argument(
        "--physical-val-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "physical" / "val",
    )
    parser.add_argument(
        "--selected-squares",
        type=str,
        default="c4,b4,g5",
        help="Comma-separated square names to show in the detailed walkthrough.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT / "two_stage_training_pipeline_cQAedm_gWrw_frame0000",
    )
    return parser


def build_context(
    *,
    annotation_id: str,
    train_root: Path,
    val_root: Path,
    selected_square_names: list[str],
) -> PipelineContext:
    split_name, row = load_annotation(annotation_id, train_root=train_root, val_root=val_root)

    clip_frame_bgr = _load_clip_frame_bgr(row, clip_cache={})
    source_kind, source_frame_bgr, training_corners = load_training_source(row)
    clip_corners = tuple((float(x), float(y)) for x, y in row.corners)

    pose = camera_pose_from_corners(
        training_corners,
        K=default_camera_matrix(source_frame_bgr.shape),
    )
    occupancy_bboxes: list[tuple[float, float, float, float]] = []
    piece_bboxes: list[tuple[float, float, float, float]] = []
    occupancy_crops_rgb: list[np.ndarray] = []
    piece_crops_rgb_by_square: dict[int, np.ndarray] = {}
    occupied_square_indices: list[int] = []

    for square_index, class_id in enumerate(row.labels):
        row_index = square_index // 8
        col_index = square_index % 8
        occupancy_bbox = square_bbox_from_corners(
            training_corners,
            row=row_index,
            col=col_index,
            pad_ratio=0.3,
        )
        piece_bbox = piece_bbox_from_projection(
            project_piece_box(
                pose,
                row=row_index,
                col=col_index,
                piece_height=DEFAULT_PIECE_HEIGHT,
                corners=training_corners,
            )
        )
        occupancy_bboxes.append(occupancy_bbox)
        piece_bboxes.append(piece_bbox)

        occupancy_debug = axis_aligned_crop_debug(
            source_frame_bgr,
            occupancy_bbox,
            output_size=DEFAULT_OCCUPANCY_CROP_SIZE,
            flip_horizontally=False,
        )
        occupancy_crops_rgb.append(occupancy_debug.canvas_rgb)

        if int(class_id) != 0:
            occupied_square_indices.append(square_index)
            piece_debug = axis_aligned_crop_debug(
                source_frame_bgr,
                piece_bbox,
                output_size=DEFAULT_PIECE_CROP_SIZE,
                flip_horizontally=(col_index < 4),
            )
            piece_crops_rgb_by_square[square_index] = piece_debug.canvas_rgb

    selected_squares = tuple(
        build_selected_square(
            square_name=square_name,
            row=row,
            source_frame_bgr=source_frame_bgr,
            training_corners=training_corners,
            occupancy_bbox=occupancy_bboxes[square_name_to_index(square_name)],
            piece_bbox=piece_bboxes[square_name_to_index(square_name)],
        )
        for square_name in selected_square_names
    )

    return PipelineContext(
        annotation_id=annotation_id,
        split_name=split_name,
        row=row,
        source_kind=source_kind,
        source_frame_bgr=source_frame_bgr,
        clip_frame_bgr=clip_frame_bgr,
        training_corners=training_corners,
        clip_corners=clip_corners,
        occupancy_bboxes=tuple(occupancy_bboxes),
        piece_bboxes=tuple(piece_bboxes),
        occupancy_crops_rgb=tuple(occupancy_crops_rgb),
        piece_crops_rgb_by_square=piece_crops_rgb_by_square,
        occupied_square_indices=tuple(occupied_square_indices),
        selected_squares=selected_squares,
    )


def load_annotation(
    annotation_id: str,
    *,
    train_root: Path,
    val_root: Path,
) -> tuple[str, Any]:
    for split_name, root in (("train", train_root), ("val", val_root)):
        rows = load_annotated_oblique_rows(root)
        for row in rows:
            if row.annotation_id == annotation_id:
                return split_name, row
    raise ValueError(f"annotation_id not found in train/val roots: {annotation_id}")


def load_training_source(row: Any) -> tuple[str, np.ndarray, tuple[tuple[float, float], ...]]:
    has_native = (
        row.source_video_id is not None
        and row.source_frame_index is not None
        and row.native_corners is not None
        and row.native_image_bbox is not None
    )
    if not has_native:
        frame_bgr = _load_clip_frame_bgr(row, clip_cache={})
        return "clip_frame", frame_bgr, tuple((float(x), float(y)) for x, y in row.corners)

    loader = NativeFrameLoader()
    try:
        frame_bgr = loader.load(
            source_video_id=str(row.source_video_id),
            source_frame_index=int(row.source_frame_index),
        )
    finally:
        loader.close()
    x_off, y_off, _width, _height = row.native_image_bbox
    full_corners = tuple((float(x + x_off), float(y + y_off)) for x, y in row.native_corners)
    return "native_frame", frame_bgr, full_corners


def build_selected_square(
    *,
    square_name: str,
    row: Any,
    source_frame_bgr: np.ndarray,
    training_corners: tuple[tuple[float, float], ...],
    occupancy_bbox: tuple[float, float, float, float],
    piece_bbox: tuple[float, float, float, float],
) -> SelectedSquare:
    square_index = square_name_to_index(square_name)
    class_id = int(row.labels[square_index])
    label_name = SQUARE_CLASS_NAMES[class_id]
    occupancy_debug = axis_aligned_crop_debug(
        source_frame_bgr,
        occupancy_bbox,
        output_size=DEFAULT_OCCUPANCY_CROP_SIZE,
        flip_horizontally=False,
    )
    col_index = square_index % 8
    piece_debug: CropDebug | None = None
    if class_id != 0:
        piece_debug = axis_aligned_crop_debug(
            source_frame_bgr,
            piece_bbox,
            output_size=DEFAULT_PIECE_CROP_SIZE,
            flip_horizontally=(col_index < 4),
        )
    return SelectedSquare(
        square_name=square_name,
        square_index=square_index,
        label_name=label_name,
        occupancy_label_name="occupied" if square_class_to_occupancy_label(class_id) else "empty",
        piece_label_name=(SQUARE_CLASS_NAMES[class_id] if class_id != 0 else None),
        flip_piece_crop=(class_id != 0 and col_index < 4),
        occupancy_bbox=occupancy_bbox,
        piece_bbox=(piece_bbox if class_id != 0 else None),
        occupancy_raw_crop_rgb=occupancy_debug.raw_crop_rgb,
        occupancy_canvas_rgb=occupancy_debug.canvas_rgb,
        piece_raw_crop_rgb=(None if piece_debug is None else piece_debug.raw_crop_rgb),
        piece_canvas_rgb=(None if piece_debug is None else piece_debug.canvas_rgb),
    )


def axis_aligned_crop_debug(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    output_size: int,
    flip_horizontally: bool,
) -> CropDebug:
    xmin, ymin, xmax, ymax = bbox
    height, width = image_bgr.shape[:2]
    clipped_xmin = max(0, int(np.floor(xmin)))
    clipped_ymin = max(0, int(np.floor(ymin)))
    clipped_xmax = min(width, int(np.ceil(xmax)))
    clipped_ymax = min(height, int(np.ceil(ymax)))

    if clipped_xmax <= clipped_xmin or clipped_ymax <= clipped_ymin:
        empty = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        return CropDebug(
            raw_crop_rgb=empty,
            canvas_rgb=empty,
            clipped_bbox=(clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax),
            resized_size=(0, 0),
            scale=0.0,
        )

    crop_bgr = image_bgr[clipped_ymin:clipped_ymax, clipped_xmin:clipped_xmax]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_h, crop_w = crop_rgb.shape[:2]
    scale = float(output_size) / float(max(crop_h, crop_w))
    new_h = max(1, int(round(crop_h * scale)))
    new_w = max(1, int(round(crop_w * scale)))
    interpolation = cv2.INTER_AREA if max(crop_h, crop_w) >= output_size else cv2.INTER_LINEAR
    resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=interpolation)
    if flip_horizontally:
        resized = cv2.flip(resized, 1)

    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    y_offset = output_size - new_h
    canvas[y_offset : y_offset + new_h, 0:new_w] = resized
    return CropDebug(
        raw_crop_rgb=crop_rgb,
        canvas_rgb=canvas,
        clipped_bbox=(clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax),
        resized_size=(new_w, new_h),
        scale=scale,
    )


def write_outputs(context: PipelineContext, *, output_dir: Path) -> None:
    _save_rgb(output_dir / "01_input_sources.png", render_input_sources(context))
    _save_rgb(output_dir / "02_geometry_overlay.png", render_geometry_overlay(context))
    _save_rgb(
        output_dir / "03_occupancy_contact_sheet.png",
        render_occupancy_contact_sheet(context),
    )
    _save_rgb(
        output_dir / "04_piece_contact_sheet.png",
        render_piece_contact_sheet(context),
    )
    _save_rgb(output_dir / "05_selected_square_examples.png", render_selected_examples(context))
    write_summary(context, output_dir=output_dir)


def render_input_sources(context: PipelineContext) -> np.ndarray:
    native_panel = draw_frame_with_corners(
        context.source_frame_bgr,
        corners=context.training_corners,
        title=(
            f"training source: {context.source_kind} | {context.source_frame_bgr.shape[1]}x"
            f"{context.source_frame_bgr.shape[0]}"
        ),
        extra_lines=[
            "this is the frame the two-stage datasets crop from",
            f"annotation: {context.annotation_id}",
        ],
    )
    clip_panel = draw_frame_with_corners(
        context.clip_frame_bgr,
        corners=context.clip_corners,
        title=(
            f"stored clip frame | {context.clip_frame_bgr.shape[1]}x"
            f"{context.clip_frame_bgr.shape[0]}"
        ),
        extra_lines=[
            "available in annotations, but not the default source when native metadata exists",
            f"clip path: {Path(context.row.clip_path).name}",
        ],
    )
    return stack_horizontal([native_panel, clip_panel], gap=12, bg=(20, 20, 20))


def render_geometry_overlay(context: PipelineContext) -> np.ndarray:
    image_rgb = cv2.cvtColor(context.source_frame_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()

    for square_index in range(64):
        row_index = square_index // 8
        col_index = square_index % 8
        quad = project_square_base_quad(context.training_corners, row=row_index, col=col_index)
        cv2.polylines(
            overlay,
            [quad.astype(np.int32)],
            isClosed=True,
            color=_BOARD_COLOR,
            thickness=1,
        )
        x1, y1, x2, y2 = (int(round(value)) for value in context.occupancy_bboxes[square_index])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _OCCUPANCY_COLOR, 1)
        if square_index in context.occupied_square_indices:
            px1, py1, px2, py2 = (int(round(value)) for value in context.piece_bboxes[square_index])
            cv2.rectangle(overlay, (px1, py1), (px2, py2), _PIECE_COLOR, 1)

    for selected in context.selected_squares:
        color = _SELECTED_COLORS.get(selected.square_name, (255, 255, 255))
        quad = project_square_base_quad(
            context.training_corners,
            row=selected.square_index // 8,
            col=selected.square_index % 8,
        )
        cv2.polylines(overlay, [quad.astype(np.int32)], True, color, 4)
        center = quad.mean(axis=0).astype(int)
        cv2.putText(
            overlay,
            selected.square_name,
            (int(center[0]) - 18, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

    for index, point in enumerate(context.training_corners):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(overlay, (x, y), 6, _CORNER_COLOR, -1)
        cv2.putText(
            overlay,
            f"c{index}",
            (x + 8, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _CORNER_COLOR,
            1,
            cv2.LINE_AA,
        )

    return annotate_image(
        overlay,
        [
            "green = projected square base quad",
            "cyan = occupancy bbox (all 64 squares)",
            "orange = piece bbox (occupied squares only)",
        ],
        top_padding=66,
    )


def render_occupancy_contact_sheet(context: PipelineContext) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for square_index in range(64):
        label_name = SQUARE_CLASS_NAMES[int(context.row.labels[square_index])]
        occupancy_label = "occupied" if int(context.row.labels[square_index]) else "empty"
        square_name = index_to_square_name(square_index)
        tile = annotate_image(
            context.occupancy_crops_rgb[square_index],
            [f"{square_name} | {occupancy_label} | raw {label_name}"],
            font_size=14,
            top_padding=24,
        )
        tiles.append(tile)
    grid = tile_grid(tiles, columns=8, gap=8, bg=(22, 22, 22))
    return annotate_image(
        grid,
        [
            "occupancy dataset samples | one sample per square | output size 112x112",
            "label = empty/occupied; every board contributes exactly 64 occupancy samples",
        ],
        top_padding=48,
    )


def render_piece_contact_sheet(context: PipelineContext) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for square_index in context.occupied_square_indices:
        square_name = index_to_square_name(square_index)
        label_name = SQUARE_CLASS_NAMES[int(context.row.labels[square_index])]
        piece_label = square_class_to_piece_label(int(context.row.labels[square_index]))
        col_index = square_index % 8
        flip_text = "flip" if col_index < 4 else "no-flip"
        tile = annotate_image(
            context.piece_crops_rgb_by_square[square_index],
            [f"{square_name} | {label_name} | piece_label={piece_label} | {flip_text}"],
            font_size=14,
            top_padding=24,
        )
        tiles.append(tile)
    grid = tile_grid(tiles, columns=6, gap=8, bg=(22, 22, 22))
    return annotate_image(
        grid,
        [
            "piece dataset samples | occupied squares only | output size 224x224",
            "label = square_class - 1; empties are skipped entirely for the piece classifier",
        ],
        top_padding=48,
    )


def render_selected_examples(context: PipelineContext) -> np.ndarray:
    rows: list[np.ndarray] = []
    for selected in context.selected_squares:
        rows.append(render_selected_square_row(context, selected))
    return stack_vertical(rows, gap=18, bg=(18, 18, 18))


def render_selected_square_row(context: PipelineContext, selected: SelectedSquare) -> np.ndarray:
    highlight = render_selected_square_highlight(context, selected)

    occupancy_panel = annotate_image(
        stack_horizontal(
            [
                _fit_panel(selected.occupancy_raw_crop_rgb, 160),
                _fit_panel(selected.occupancy_canvas_rgb, 160),
            ],
            gap=12,
            bg=(18, 18, 18),
        ),
        [
            (
                f"occupancy branch | label={selected.occupancy_label_name} | "
                f"bbox={format_bbox(selected.occupancy_bbox)}"
            ),
            "raw clipped bbox -> resized/pasted to 112x112 canvas",
        ],
        top_padding=48,
    )

    if selected.piece_label_name is None:
        piece_panel = text_panel(
            title="piece branch",
            lines=[
                "no piece sample for this square",
                "reason: piece dataset skips empty squares",
            ],
            size=(360, occupancy_panel.shape[0]),
        )
    else:
        piece_panel = annotate_image(
            stack_horizontal(
                [
                    _fit_panel(selected.piece_raw_crop_rgb, 224),
                    _fit_panel(selected.piece_canvas_rgb, 224),
                ],
                gap=12,
                bg=(18, 18, 18),
            ),
            [
                (
                    f"piece branch | label={selected.piece_label_name} | "
                    f"bbox={format_bbox(selected.piece_bbox)}"
                ),
                (
                    "raw clipped bbox -> resized/pasted to 224x224 canvas | "
                    f"{'flipped horizontally' if selected.flip_piece_crop else 'kept as-is'}"
                ),
            ],
            top_padding=48,
        )

    row = stack_horizontal([highlight, occupancy_panel, piece_panel], gap=12, bg=(18, 18, 18))
    return annotate_image(
        row,
        [
            (
                f"square {selected.square_name} | raw class={selected.label_name} | "
                f"occupancy={selected.occupancy_label_name}"
            )
        ],
        font_size=22,
        top_padding=42,
    )


def render_selected_square_highlight(
    context: PipelineContext,
    selected: SelectedSquare,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(context.source_frame_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()
    quad = project_square_base_quad(
        context.training_corners,
        row=selected.square_index // 8,
        col=selected.square_index % 8,
    )
    color = _SELECTED_COLORS.get(selected.square_name, (255, 255, 255))
    cv2.polylines(overlay, [quad.astype(np.int32)], True, color, 5)
    ox1, oy1, ox2, oy2 = (int(round(value)) for value in selected.occupancy_bbox)
    cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), _OCCUPANCY_COLOR, 3)
    if selected.piece_bbox is not None:
        px1, py1, px2, py2 = (int(round(value)) for value in selected.piece_bbox)
        cv2.rectangle(overlay, (px1, py1), (px2, py2), _PIECE_COLOR, 3)
    return annotate_image(
        overlay,
        [
            f"source frame highlight | {selected.square_name}",
            "green = board square | cyan = occupancy bbox | orange = piece bbox",
        ],
        top_padding=48,
    )


def write_summary(context: PipelineContext, *, output_dir: Path) -> None:
    summary_path = output_dir / "summary.md"
    stats_path = output_dir / "stats.json"

    occupancy_sizes = [bbox_size(bbox) for bbox in context.occupancy_bboxes]
    piece_sizes = [
        bbox_size(context.piece_bboxes[index]) for index in context.occupied_square_indices
    ]
    stats = {
        "annotation_id": context.annotation_id,
        "split": context.split_name,
        "source_kind": context.source_kind,
        "source_frame_shape": list(context.source_frame_bgr.shape),
        "clip_frame_shape": list(context.clip_frame_bgr.shape),
        "occupied_square_count": len(context.occupied_square_indices),
        "empty_square_count": 64 - len(context.occupied_square_indices),
        "occupancy_bbox_size_summary": summarize_sizes(occupancy_sizes),
        "piece_bbox_size_summary": summarize_sizes(piece_sizes),
        "selected_squares": {
            selected.square_name: {
                "label_name": selected.label_name,
                "occupancy_label_name": selected.occupancy_label_name,
                "piece_label_name": selected.piece_label_name,
                "flip_piece_crop": selected.flip_piece_crop,
                "occupancy_bbox": list(selected.occupancy_bbox),
                "piece_bbox": None if selected.piece_bbox is None else list(selected.piece_bbox),
            }
            for selected in context.selected_squares
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))

    lines = [
        "# Two-stage training pipeline walkthrough",
        "",
        f"Annotation: `{context.annotation_id}` ({context.split_name})",
        f"Training source used by dataset: `{context.source_kind}`",
        "",
        "## Code path",
        "",
        "- dataset construction: `pipeline/physical/two_stage/classifier_data.py`",
        "- geometry: `pipeline/physical/piece_projection.py`",
        "- training: `scripts/train_square_classifier.py`",
        "- runtime composition: `pipeline/physical/two_stage/reader.py`",
        "",
        "## What happens for this single board frame",
        "",
        "1. Load the annotated board row from `board_annotations.jsonl`.",
        (
            "2. If native metadata exists, load the **native source-video frame** "
            "and lift the corners into full-frame coordinates."
        ),
        (
            "3. For each of the 64 squares, compute an **occupancy bbox** from "
            "the projected square base quad with `pad_ratio=0.3`."
        ),
        (
            "4. For each occupied square only, compute a **piece bbox** from the "
            "projected 3D piece box (`piece_height=2.0`)."
        ),
        (
            "5. Extract axis-aligned crops, resize them to a fixed square canvas, "
            "and bottom-align them."
        ),
        "6. For left-half piece crops (`col < 4`), flip horizontally before training.",
        (
            "7. Convert crops to RGB tensors, optionally augment (`--augment`), "
            "then ImageNet-normalize."
        ),
        "8. Feed each crop independently through the frozen vision encoder and a linear head.",
        "",
        "## Label mapping",
        "",
        "- occupancy task: `0=empty`, `1=occupied`",
        "- piece task: `piece_label = square_class - 1`",
        "- empty squares are **not** part of the piece dataset",
        "",
        "## Per-frame counts",
        "",
        "- occupancy samples from this board: `64`",
        f"- piece samples from this board: `{len(context.occupied_square_indices)}`",
        f"- empty squares skipped by piece dataset: `{64 - len(context.occupied_square_indices)}`",
        "",
        "## Bounding-box size summary on this frame",
        "",
        bbox_summary_line("occupancy", occupancy_sizes),
        bbox_summary_line("piece (occupied only)", piece_sizes),
        "",
        "## Files",
        "",
        "- `01_input_sources.png` — native training source vs stored clip frame",
        "- `02_geometry_overlay.png` — square quads, occupancy bboxes, and occupied-piece bboxes",
        "- `03_occupancy_contact_sheet.png` — all 64 occupancy samples",
        "- `04_piece_contact_sheet.png` — occupied piece samples only",
        "- `05_selected_square_examples.png` — detailed step-by-step examples",
        "- `stats.json` — raw numbers",
        "",
        "## Important details that can explain failures",
        "",
        "1. The two-stage path is **crop-based**, not whole-board token pooling.",
        "2. Occupancy and piece branches do **not** see the same geometry.",
        "3. Piece crops on files `a`-`d` are flipped horizontally; files `e`-`h` are not.",
        "4. Empty squares train the occupancy model but never train the piece model.",
        (
            "5. The train script defaults to a frozen encoder; only the final "
            "linear head is trained unless you unfreeze layers."
        ),
        "6. Augmentation is off unless `scripts/train_square_classifier.py --augment` is used.",
        "",
        "## Selected examples on this frame",
        "",
    ]

    for selected in context.selected_squares:
        lines.extend(
            [
                f"### {selected.square_name}",
                f"- raw square class: `{selected.label_name}`",
                f"- occupancy label: `{selected.occupancy_label_name}`",
                (
                    f"- piece label: `{selected.piece_label_name}`"
                    if selected.piece_label_name is not None
                    else "- piece label: `none` (empty squares are skipped)"
                ),
                f"- occupancy bbox: `{format_bbox(selected.occupancy_bbox)}`",
                (
                    f"- piece bbox: `{format_bbox(selected.piece_bbox)}`"
                    if selected.piece_bbox is not None
                    else "- piece bbox: `not used`"
                ),
                (
                    f"- piece crop flip: `{'yes' if selected.flip_piece_crop else 'no'}`"
                    if selected.piece_label_name is not None
                    else "- piece crop flip: `n/a`"
                ),
                "",
            ]
        )

    summary_path.write_text("\n".join(lines) + "\n")


def summarize_sizes(sizes: list[tuple[float, float]]) -> dict[str, float]:
    widths = np.array([size[0] for size in sizes], dtype=np.float64)
    heights = np.array([size[1] for size in sizes], dtype=np.float64)
    areas = widths * heights
    return {
        "width_min": float(widths.min()),
        "width_mean": float(widths.mean()),
        "width_max": float(widths.max()),
        "height_min": float(heights.min()),
        "height_mean": float(heights.mean()),
        "height_max": float(heights.max()),
        "area_min": float(areas.min()),
        "area_mean": float(areas.mean()),
        "area_max": float(areas.max()),
    }


def bbox_summary_line(name: str, sizes: list[tuple[float, float]]) -> str:
    summary = summarize_sizes(sizes)
    return (
        f"- {name}: width `{summary['width_min']:.1f}..{summary['width_max']:.1f}` "
        f"(mean `{summary['width_mean']:.1f}`), height `{summary['height_min']:.1f}.."
        f"{summary['height_max']:.1f}` (mean `{summary['height_mean']:.1f}`), area mean "
        f"`{summary['area_mean']:.1f}` px²"
    )


def bbox_size(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def draw_frame_with_corners(
    image_bgr: np.ndarray,
    *,
    corners: tuple[tuple[float, float], ...],
    title: str,
    extra_lines: list[str],
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()
    for index, point in enumerate(corners):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(overlay, (x, y), 6, _CORNER_COLOR, -1)
        cv2.putText(
            overlay,
            f"c{index}",
            (x + 8, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _CORNER_COLOR,
            2,
            cv2.LINE_AA,
        )
    return annotate_image(overlay, [title, *extra_lines], top_padding=68)


def annotate_image(
    image_rgb: np.ndarray,
    lines: list[str],
    *,
    font_size: int = 16,
    top_padding: int = 32,
) -> np.ndarray:
    font = load_font(font_size)
    width = image_rgb.shape[1]
    caption = Image.new("RGB", (width, top_padding), (24, 24, 24))
    draw = ImageDraw.Draw(caption)
    y = 4
    for line in lines:
        draw.text((8, y), line, fill=(245, 245, 245), font=font)
        y += font_size + 2
    return stack_vertical([np.asarray(caption), image_rgb], gap=0, bg=(24, 24, 24))


def text_panel(*, title: str, lines: list[str], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    panel = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(panel)
    title_font = load_font(20)
    body_font = load_font(16)
    draw.text((12, 12), title, fill=(245, 245, 245), font=title_font)
    y = 52
    for line in lines:
        draw.text((12, y), line, fill=(230, 230, 230), font=body_font)
        y += 24
    return np.asarray(panel)


def tile_grid(
    images: list[np.ndarray],
    *,
    columns: int,
    gap: int,
    bg: tuple[int, int, int],
) -> np.ndarray:
    rows = math.ceil(len(images) / columns)
    tile_width = max(image.shape[1] for image in images)
    tile_height = max(image.shape[0] for image in images)
    canvas = np.full(
        (
            rows * tile_height + max(0, rows - 1) * gap,
            columns * tile_width + max(0, columns - 1) * gap,
            3,
        ),
        bg,
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x = col * (tile_width + gap)
        y = row * (tile_height + gap)
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    return canvas


def stack_horizontal(
    images: list[np.ndarray],
    *,
    gap: int,
    bg: tuple[int, int, int],
) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * max(0, len(images) - 1)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    x = 0
    for image in images:
        y = (height - image.shape[0]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return canvas


def stack_vertical(
    images: list[np.ndarray],
    *,
    gap: int,
    bg: tuple[int, int, int],
) -> np.ndarray:
    width = max(image.shape[1] for image in images)
    height = sum(image.shape[0] for image in images) + gap * max(0, len(images) - 1)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    y = 0
    for image in images:
        x = (width - image.shape[1]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        y += image.shape[0] + gap
    return canvas


def _fit_panel(image_rgb: np.ndarray | None, max_size: int) -> np.ndarray:
    if image_rgb is None:
        return np.full((max_size, max_size, 3), (0, 0, 0), dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    scale = float(max_size) / float(max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((max_size, max_size, 3), (0, 0, 0), dtype=np.uint8)
    x = (max_size - new_w) // 2
    y = (max_size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def format_bbox(bbox: tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return "n/a"
    return ", ".join(f"{value:.1f}" for value in bbox)


def square_name_to_index(square_name: str) -> int:
    file_index = ord(square_name[0].lower()) - ord("a")
    rank = int(square_name[1])
    return (8 - rank) * 8 + file_index


def index_to_square_name(square_index: int) -> str:
    row_index, col_index = divmod(square_index, 8)
    return f"{chr(ord('a') + col_index)}{8 - row_index}"


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


if __name__ == "__main__":
    main()
