"""Runtime visualizations for the held-out physical-board eval set."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.physical.board_data import PhysicalEvalBoardDataset, PhysicalEvalBoardRow
from pipeline.physical.square_classifier import (
    PhysicalBoardSequenceReader,
    read_board_observation_from_frame,
)
from pipeline.shared import SQUARE_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EMPTY_CLASS_ID = 0
_LIGHT_SQUARE = (240, 217, 181)
_DARK_SQUARE = (181, 136, 99)
_CORRECT_COLOR = (51, 171, 88)
_ERROR_COLOR = (204, 65, 65)
_GT_CHANGED_BORDER = (244, 208, 63)
_PRED_CHANGED_BORDER = (73, 160, 255)
_TEXT_COLOR = (24, 24, 24)
_GRID_COLOR = (40, 40, 40)
_HEADER_BG = (248, 248, 248)
_PANEL_BG = (255, 255, 255)
_CONTACT_SHEET_BG = (245, 245, 245)

_CLASS_ID_TO_SYMBOL = {
    class_id: ("" if name == "empty" else name)
    for class_id, name in enumerate(SQUARE_CLASS_NAMES)
}


@dataclass(frozen=True)
class VisualizedRuntimeFrame:
    frame_index: int
    annotation_id: str
    board_path: str
    crop_rgb: np.ndarray
    gt_class_ids: tuple[int, ...]
    stateless_class_ids: tuple[int, ...]
    temporal_class_ids: tuple[int, ...]
    stateless_confidences: tuple[float, ...]
    temporal_confidences: tuple[float, ...]
    stateless_error_count: int
    temporal_error_count: int
    gt_change_count: int | None
    stateless_change_count: int | None
    temporal_change_count: int | None
    gt_changed_mask: tuple[bool, ...]
    stateless_changed_mask: tuple[bool, ...]
    temporal_changed_mask: tuple[bool, ...]

    @property
    def stateless_mean_confidence(self) -> float:
        return float(sum(self.stateless_confidences) / len(self.stateless_confidences))

    @property
    def temporal_mean_confidence(self) -> float:
        return float(sum(self.temporal_confidences) / len(self.temporal_confidences))


def visualize_runtime_sequence(
    *,
    clip_path: str | None,
    frame_start: int,
    frame_count: int,
    device: str,
    output_dir: str | Path,
    panel_size: int = 240,
) -> dict[str, Any]:
    """Render a frame-by-frame runtime visualization for one held-out eval clip."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    if panel_size <= 0:
        raise ValueError(f"panel_size must be > 0, got {panel_size}")

    dataset = PhysicalEvalBoardDataset()
    rows_by_clip = _group_rows_by_clip(dataset.rows)
    selected_clip_path = _select_clip_path(rows_by_clip, clip_path=clip_path)
    clip_rows = rows_by_clip[selected_clip_path]
    visualized_frames = _collect_visualized_frames(
        clip_rows,
        frame_start=frame_start,
        frame_count=frame_count,
        device=device,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_images: list[Image.Image] = []
    frame_manifest: list[dict[str, Any]] = []
    for frame in visualized_frames:
        rendered = render_visualized_runtime_frame(frame, panel_size=panel_size)
        frame_filename = f"frame_{frame.frame_index:04d}.png"
        frame_path = frame_dir / frame_filename
        rendered.save(frame_path)
        frame_images.append(rendered)
        frame_manifest.append(_frame_manifest_row(frame, image_path=frame_path))

    contact_sheet = render_contact_sheet(
        frame_images,
        clip_path=selected_clip_path,
        frame_start=frame_start,
        frame_count=len(visualized_frames),
    )
    contact_sheet_path = output_path / "contact_sheet.png"
    contact_sheet.save(contact_sheet_path)

    summary = {
        "clip_path": selected_clip_path,
        "frame_start": frame_start,
        "frame_count": len(visualized_frames),
        "available_frame_count": len(clip_rows),
        "contact_sheet": _relative_to_project(contact_sheet_path),
        "frames": frame_manifest,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def render_visualized_runtime_frame(
    frame: VisualizedRuntimeFrame,
    *,
    panel_size: int,
) -> Image.Image:
    """Render one frame row with crop, ground truth, and runtime predictions."""
    margin = 12
    panel_gap = 12
    header_height = 44
    panel_labels_height = 22
    width = margin * 2 + panel_size * 4 + panel_gap * 3
    height = margin * 2 + header_height + panel_labels_height + panel_size

    image = Image.new("RGB", (width, height), _PANEL_BG)
    draw = ImageDraw.Draw(image)
    header_font = _load_font(20)
    body_font = _load_font(15)

    header_text = (
        f"frame {frame.frame_index:04d} | gtΔ={_fmt_delta(frame.gt_change_count)} "
        f"singleΔ={_fmt_delta(frame.stateless_change_count)} "
        f"tempΔ={_fmt_delta(frame.temporal_change_count)}"
    )
    metrics_text = (
        f"single err={frame.stateless_error_count:02d} conf={frame.stateless_mean_confidence:.2f}"
        f" | temp err={frame.temporal_error_count:02d} conf={frame.temporal_mean_confidence:.2f}"
    )
    draw.rectangle((0, 0, width, margin + header_height), fill=_HEADER_BG)
    draw.text((margin, margin), header_text, fill=_TEXT_COLOR, font=header_font)
    draw.text((margin, margin + 22), metrics_text, fill=_TEXT_COLOR, font=body_font)

    top = margin + header_height + panel_labels_height
    lefts = [margin + (panel_size + panel_gap) * index for index in range(4)]
    labels = ["crop + temporal errors", "ground truth", "stateless", "temporal"]
    panels = [
        _render_crop_panel(
            frame,
            panel_size=panel_size,
            gt_changed_mask=frame.gt_changed_mask,
            temporal_changed_mask=frame.temporal_changed_mask,
        ),
        _render_board_panel(
            frame.gt_class_ids,
            panel_size=panel_size,
            confidences=None,
            target_class_ids=frame.gt_class_ids,
            explicit_changed_mask=frame.gt_changed_mask,
            border_color=_GT_CHANGED_BORDER,
        ),
        _render_board_panel(
            frame.stateless_class_ids,
            panel_size=panel_size,
            confidences=frame.stateless_confidences,
            target_class_ids=frame.gt_class_ids,
            explicit_changed_mask=frame.stateless_changed_mask,
            border_color=_PRED_CHANGED_BORDER,
        ),
        _render_board_panel(
            frame.temporal_class_ids,
            panel_size=panel_size,
            confidences=frame.temporal_confidences,
            target_class_ids=frame.gt_class_ids,
            explicit_changed_mask=frame.temporal_changed_mask,
            border_color=_PRED_CHANGED_BORDER,
        ),
    ]

    for left, label, panel in zip(lefts, labels, panels):
        draw.text((left, margin + header_height), label, fill=_TEXT_COLOR, font=body_font)
        image.paste(panel, (left, top))

    return image


def render_contact_sheet(
    frame_images: list[Image.Image],
    *,
    clip_path: str,
    frame_start: int,
    frame_count: int,
) -> Image.Image:
    """Stack rendered frame rows into one readable contact sheet."""
    if not frame_images:
        raise ValueError("frame_images must be non-empty")

    width = max(image.width for image in frame_images)
    title_height = 72
    total_height = title_height + sum(image.height for image in frame_images)
    contact_sheet = Image.new("RGB", (width, total_height), _CONTACT_SHEET_BG)
    draw = ImageDraw.Draw(contact_sheet)
    title_font = _load_font(24)
    body_font = _load_font(16)

    draw.rectangle((0, 0, width, title_height), fill=_HEADER_BG)
    title_text = f"physical runtime visualization | {Path(clip_path).name}"
    draw.text((12, 10), title_text, fill=_TEXT_COLOR, font=title_font)
    draw.text(
        (12, 40),
        (
            f"frames {frame_start}..{frame_start + frame_count - 1} | "
            "crop: red=temporal error, yellow=GT changed, blue=temporal-only flip | "
            "pred boards: green=correct, red=wrong"
        ),
        fill=_TEXT_COLOR,
        font=body_font,
    )

    top = title_height
    for frame_image in frame_images:
        contact_sheet.paste(frame_image, (0, top))
        top += frame_image.height
    return contact_sheet


def _collect_visualized_frames(
    rows: list[PhysicalEvalBoardRow],
    *,
    frame_start: int,
    frame_count: int,
    device: str,
) -> list[VisualizedRuntimeFrame]:
    selected_rows = [
        row
        for row in rows
        if row.frame_index is not None
        and frame_start <= int(row.frame_index) < frame_start + frame_count
    ]
    if not selected_rows:
        end_frame = frame_start + frame_count - 1
        raise ValueError(f"No annotated eval rows found for frames {frame_start}..{end_frame}")

    sequence_reader = PhysicalBoardSequenceReader(device=device)
    visualized_frames: list[VisualizedRuntimeFrame] = []
    previous_gt: tuple[int, ...] | None = None
    previous_stateless: tuple[int, ...] | None = None
    previous_temporal: tuple[int, ...] | None = None

    selected_frame_indices = {
        int(row.frame_index)
        for row in selected_rows
        if row.frame_index is not None
    }

    for row in rows:
        image_bgr = _load_board_image(row)
        stateless_observation = read_board_observation_from_frame(image_bgr, device=device)
        temporal_observation = sequence_reader.read_board_observation_from_frame(image_bgr)
        if stateless_observation is None or temporal_observation is None:
            raise ValueError(f"Runtime reader failed for {row.board_path}")

        gt_class_ids = tuple(int(value) for value in row.labels)
        stateless_class_ids = tuple(_fen_to_class_ids(stateless_observation.fen))
        temporal_class_ids = tuple(_fen_to_class_ids(temporal_observation.fen))

        if row.frame_index in selected_frame_indices:
            crop_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            visualized_frames.append(
                VisualizedRuntimeFrame(
                    frame_index=int(row.frame_index or 0),
                    annotation_id=row.annotation_id,
                    board_path=row.board_path,
                    crop_rgb=crop_rgb,
                    gt_class_ids=gt_class_ids,
                    stateless_class_ids=stateless_class_ids,
                    temporal_class_ids=temporal_class_ids,
                    stateless_confidences=tuple(stateless_observation.square_confidences),
                    temporal_confidences=tuple(temporal_observation.square_confidences),
                    stateless_error_count=_count_differences(stateless_class_ids, gt_class_ids),
                    temporal_error_count=_count_differences(temporal_class_ids, gt_class_ids),
                    gt_change_count=(
                        None
                        if previous_gt is None
                        else _count_differences(previous_gt, gt_class_ids)
                    ),
                    stateless_change_count=(
                        None
                        if previous_stateless is None
                        else _count_differences(previous_stateless, stateless_class_ids)
                    ),
                    temporal_change_count=(
                        None
                        if previous_temporal is None
                        else _count_differences(previous_temporal, temporal_class_ids)
                    ),
                    gt_changed_mask=_changed_mask(gt_class_ids, previous_gt),
                    stateless_changed_mask=_changed_mask(
                        stateless_class_ids,
                        previous_stateless,
                    ),
                    temporal_changed_mask=_changed_mask(
                        temporal_class_ids,
                        previous_temporal,
                    ),
                )
            )

        previous_gt = gt_class_ids
        previous_stateless = stateless_class_ids
        previous_temporal = temporal_class_ids

    return visualized_frames


def _render_crop_panel(
    frame: VisualizedRuntimeFrame,
    *,
    panel_size: int,
    gt_changed_mask: tuple[bool, ...] | None = None,
    temporal_changed_mask: tuple[bool, ...] | None = None,
) -> Image.Image:
    image = Image.fromarray(frame.crop_rgb).resize(
        (panel_size, panel_size),
        Image.Resampling.BILINEAR,
    )
    overlay = Image.new("RGBA", (panel_size, panel_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cell = panel_size / 8.0

    if gt_changed_mask is None:
        gt_changed_mask = (False,) * 64
    if temporal_changed_mask is None:
        temporal_changed_mask = (False,) * 64

    for square_index, (target, predicted, confidence) in enumerate(
        zip(frame.gt_class_ids, frame.temporal_class_ids, frame.temporal_confidences)
    ):
        row = square_index // 8
        col = square_index % 8
        x0 = int(round(col * cell))
        y0 = int(round(row * cell))
        x1 = int(round((col + 1) * cell))
        y1 = int(round((row + 1) * cell))

        if target != predicted:
            alpha = 40 + int(110 * float(confidence))
            draw.rectangle((x0, y0, x1, y1), fill=(*_ERROR_COLOR, alpha))

        if gt_changed_mask[square_index]:
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=_GT_CHANGED_BORDER, width=3)
        elif temporal_changed_mask[square_index]:
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=_PRED_CHANGED_BORDER, width=2)

    draw_grid = ImageDraw.Draw(overlay)
    for line_index in range(9):
        x = int(round(line_index * cell))
        y = int(round(line_index * cell))
        draw_grid.line((x, 0, x, panel_size), fill=(*_GRID_COLOR, 180), width=1)
        draw_grid.line((0, y, panel_size, y), fill=(*_GRID_COLOR, 180), width=1)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _render_board_panel(
    class_ids: tuple[int, ...],
    *,
    panel_size: int,
    confidences: tuple[float, ...] | None,
    target_class_ids: tuple[int, ...],
    explicit_changed_mask: tuple[bool, ...] | None,
    border_color: tuple[int, int, int],
) -> Image.Image:
    image = Image.new("RGB", (panel_size, panel_size), _PANEL_BG)
    draw = ImageDraw.Draw(image)
    cell = panel_size / 8.0
    symbol_font = _load_font(max(14, int(cell * 0.55)))

    changed_mask = explicit_changed_mask or (False,) * 64
    confidence_values = confidences or (1.0,) * 64

    for square_index, (class_id, target_class_id, confidence) in enumerate(
        zip(class_ids, target_class_ids, confidence_values)
    ):
        row = square_index // 8
        col = square_index % 8
        x0 = int(round(col * cell))
        y0 = int(round(row * cell))
        x1 = int(round((col + 1) * cell))
        y1 = int(round((row + 1) * cell))

        base_color = _LIGHT_SQUARE if (row + col) % 2 == 0 else _DARK_SQUARE
        if confidences is not None:
            correct = class_id == target_class_id
            tint = _CORRECT_COLOR if correct else _ERROR_COLOR
            weight = 0.20 + 0.55 * float(confidence)
            fill = _blend(base_color, tint, weight)
        else:
            fill = base_color
        draw.rectangle((x0, y0, x1, y1), fill=fill)

        if changed_mask[square_index]:
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=border_color, width=3)

        symbol = _CLASS_ID_TO_SYMBOL[class_id]
        if symbol:
            text_bbox = draw.textbbox((0, 0), symbol, font=symbol_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                (
                    x0 + (x1 - x0 - text_width) / 2,
                    y0 + (y1 - y0 - text_height) / 2 - 1,
                ),
                symbol,
                fill=_TEXT_COLOR,
                font=symbol_font,
            )

    for line_index in range(9):
        x = int(round(line_index * cell))
        y = int(round(line_index * cell))
        draw.line((x, 0, x, panel_size), fill=_GRID_COLOR, width=1)
        draw.line((0, y, panel_size, y), fill=_GRID_COLOR, width=1)
    return image


def _group_rows_by_clip(rows: list[PhysicalEvalBoardRow]) -> dict[str, list[PhysicalEvalBoardRow]]:
    grouped: dict[str, list[PhysicalEvalBoardRow]] = defaultdict(list)
    for row in rows:
        clip_key = row.clip_path or row.annotation_id
        grouped[clip_key].append(row)
    return {
        clip_key: sorted(
            clip_rows,
            key=lambda row: row.frame_index if row.frame_index is not None else -1,
        )
        for clip_key, clip_rows in grouped.items()
    }


def _select_clip_path(
    rows_by_clip: dict[str, list[PhysicalEvalBoardRow]],
    *,
    clip_path: str | None,
) -> str:
    if clip_path is not None:
        if clip_path not in rows_by_clip:
            raise ValueError(f"Unknown clip_path: {clip_path}")
        return clip_path
    return max(rows_by_clip.items(), key=lambda item: len(item[1]))[0]


def _load_board_image(row: PhysicalEvalBoardRow) -> np.ndarray:
    image = cv2.imread(str(_PROJECT_ROOT / row.board_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load board image: {row.board_path}")
    return image


def _fen_to_class_ids(fen: str) -> list[int]:
    placement = fen.split(" ", 1)[0]
    class_name_to_index = {name: index for index, name in enumerate(SQUARE_CLASS_NAMES)}
    class_ids: list[int] = []
    for rank in placement.split("/"):
        for char in rank:
            if char.isdigit():
                class_ids.extend([_EMPTY_CLASS_ID] * int(char))
            else:
                class_ids.append(class_name_to_index[char])
    if len(class_ids) != 64:
        raise ValueError(f"Expected 64 class ids from FEN, got {len(class_ids)}: {fen}")
    return class_ids


def _count_differences(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(int(left_value != right_value) for left_value, right_value in zip(left, right))


def _changed_mask(
    current: tuple[int, ...],
    previous: tuple[int, ...] | None,
) -> tuple[bool, ...]:
    if previous is None:
        return (False,) * len(current)
    return tuple(
        current_value != previous_value
        for current_value, previous_value in zip(current, previous)
    )


def _blend(
    base: tuple[int, int, int],
    tint: tuple[int, int, int],
    weight: float,
) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, weight))
    return tuple(
        int(round((1.0 - clamped) * base_value + clamped * tint_value))
        for base_value, tint_value in zip(base, tint)
    )


def _frame_manifest_row(frame: VisualizedRuntimeFrame, *, image_path: Path) -> dict[str, Any]:
    return {
        "annotation_id": frame.annotation_id,
        "board_path": frame.board_path,
        "frame_image": _relative_to_project(image_path),
        "frame_index": frame.frame_index,
        "gt_change_count": frame.gt_change_count,
        "stateless_change_count": frame.stateless_change_count,
        "stateless_error_count": frame.stateless_error_count,
        "stateless_mean_confidence": round(frame.stateless_mean_confidence, 4),
        "temporal_change_count": frame.temporal_change_count,
        "temporal_error_count": frame.temporal_error_count,
        "temporal_mean_confidence": round(frame.temporal_mean_confidence, 4),
    }


def _fmt_delta(value: int | None) -> str:
    return "-" if value is None else str(value)


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in font_candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(_PROJECT_ROOT.resolve()))
