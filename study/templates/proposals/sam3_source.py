from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pipeline.analysis.board_segmenter import _load_sam
from pipeline.analysis.config import VideoAnalysisConfig
from pipeline.physical.piece_projection import extract_board_neighborhood_crop
from study.templates.proposals.common import (
    ProposalFrame,
    SquareCropProposal,
    board_point_to_square_name,
)


@dataclass(frozen=True)
class Sam3ProposalConfig:
    confidence_threshold: float = 0.2
    board_crop_margin: float = 0.15
    prompts: tuple[str, ...] = (
        "white chess pieces",
        "black chess pieces",
        "chess pieces",
    )
    min_mask_pixels: int = 64
    max_bbox_width_fraction: float = 0.25
    max_bbox_height_fraction: float = 0.6
    crop_cell_size: int = 144


@dataclass(frozen=True)
class Sam3MaskCandidate:
    mask_local: np.ndarray
    bbox_local: tuple[int, int, int, int]
    score: float


def propose_sam3(
    frame: ProposalFrame,
    *,
    config: Sam3ProposalConfig | None = None,
) -> list[SquareCropProposal]:
    resolved_config = config or Sam3ProposalConfig()
    board_crop = extract_board_neighborhood_crop(
        frame.image_bgr,
        frame.corners,
        crop_margin=resolved_config.board_crop_margin,
    )
    candidates = _collect_prompt_candidates(board_crop.image_bgr, config=resolved_config)

    best_by_square: dict[str, SquareCropProposal] = {}
    for candidate in candidates:
        square_name = _mask_to_square_name(board_crop.corners.tolist(), candidate.mask_local)
        if square_name is None:
            continue
        proposal = _proposal_from_candidate(
            frame=frame,
            board_crop=board_crop,
            square_name=square_name,
            candidate=candidate,
        )
        current_best = best_by_square.get(square_name)
        if current_best is None or float(proposal.score or 0.0) > float(current_best.score or 0.0):
            best_by_square[square_name] = proposal

    return [best_by_square[square] for square in sorted(best_by_square)]


def render_sam3_preview(
    frame: ProposalFrame,
    proposals: list[SquareCropProposal],
    *,
    output_path: str | Path,
    config: Sam3ProposalConfig | None = None,
) -> None:
    resolved_config = config or Sam3ProposalConfig()
    board_crop = extract_board_neighborhood_crop(
        frame.image_bgr,
        frame.corners,
        crop_margin=resolved_config.board_crop_margin,
    )
    original_panel = _render_original_panel(frame, proposals)
    board_panel = _render_board_panel(board_crop.image_bgr, board_crop.x1, board_crop.y1, proposals)
    crop_grid = _render_crop_grid(proposals, cell_size=resolved_config.crop_cell_size)

    top_height = max(original_panel.shape[0], board_panel.shape[0])
    top_width = original_panel.shape[1] + board_panel.shape[1] + 16
    top_canvas = np.full((top_height, top_width, 3), 24, dtype=np.uint8)
    top_canvas[: original_panel.shape[0], : original_panel.shape[1]] = original_panel
    board_left = original_panel.shape[1] + 16
    top_canvas[: board_panel.shape[0], board_left : board_left + board_panel.shape[1]] = board_panel

    canvas_width = max(top_canvas.shape[1], crop_grid.shape[1])
    canvas_height = top_canvas.shape[0] + 16 + crop_grid.shape[0]
    canvas = np.full((canvas_height, canvas_width, 3), 24, dtype=np.uint8)
    canvas[: top_canvas.shape[0], : top_canvas.shape[1]] = top_canvas
    canvas[top_canvas.shape[0] + 16 :, : crop_grid.shape[1]] = crop_grid

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise ValueError(f"Failed to write SAM3 preview: {output}")


def _collect_prompt_candidates(
    board_image_bgr: np.ndarray,
    *,
    config: Sam3ProposalConfig,
) -> list[Sam3MaskCandidate]:
    processor = _get_sam_processor(config.confidence_threshold)
    board_image_rgb = cv2.cvtColor(board_image_bgr, cv2.COLOR_BGR2RGB)
    state = processor.set_image(Image.fromarray(board_image_rgb))

    candidates: list[Sam3MaskCandidate] = []
    for prompt in config.prompts:
        prompt_state = processor.set_text_prompt(prompt, _clone_sam_state(state))
        masks = np.asarray(prompt_state.get("masks"))
        scores = np.asarray(prompt_state.get("scores"))
        if masks.size == 0 or scores.size == 0:
            continue
        for mask, score in zip(masks, scores, strict=False):
            mask_local = np.asarray(mask).squeeze().astype(bool)
            if int(mask_local.sum()) < config.min_mask_pixels:
                continue
            bbox_local = _mask_bbox(mask_local)
            if bbox_local is None:
                continue
            if _bbox_is_too_large(
                bbox_local,
                image_shape=board_image_bgr.shape,
                max_width_fraction=config.max_bbox_width_fraction,
                max_height_fraction=config.max_bbox_height_fraction,
            ):
                continue
            candidates.append(
                Sam3MaskCandidate(
                    mask_local=mask_local,
                    bbox_local=bbox_local,
                    score=float(score),
                )
            )
    return candidates


def _proposal_from_candidate(
    *,
    frame: ProposalFrame,
    board_crop: Any,
    square_name: str,
    candidate: Sam3MaskCandidate,
) -> SquareCropProposal:
    x1_local, y1_local, x2_local, y2_local = candidate.bbox_local
    crop_bgr = board_crop.image_bgr[y1_local:y2_local, x1_local:x2_local].copy()

    full_mask = np.zeros(frame.image_bgr.shape[:2], dtype=np.uint8)
    crop_height, crop_width = board_crop.image_bgr.shape[:2]
    full_mask[
        board_crop.y1 : board_crop.y1 + crop_height,
        board_crop.x1 : board_crop.x1 + crop_width,
    ] = candidate.mask_local.astype(np.uint8) * 255

    x1_full = int(board_crop.x1 + x1_local)
    y1_full = int(board_crop.y1 + y1_local)
    x2_full = int(board_crop.x1 + x2_local)
    y2_full = int(board_crop.y1 + y2_local)
    return SquareCropProposal(
        square=square_name,
        crop_bgr=crop_bgr,
        score=candidate.score,
        bbox=(x1_full, y1_full, x2_full, y2_full),
        mask=full_mask,
    )


def _mask_to_square_name(
    corners: list[list[float]] | tuple[tuple[float, float], ...],
    mask_local: np.ndarray,
) -> str | None:
    if mask_local.ndim != 2:
        raise ValueError(f"mask_local must be 2D, got shape {mask_local.shape}")
    ys, xs = np.nonzero(mask_local)
    if len(xs) == 0:
        return None

    y_threshold = np.quantile(ys.astype(np.float32), 0.75)
    bottom_band = ys >= y_threshold
    if bottom_band.any():
        probe_x = float(xs[bottom_band].mean())
        probe_y = float(ys[bottom_band].mean())
    else:
        probe_x = float(xs.mean())
        probe_y = float(ys.mean())

    board_polygon = np.asarray(corners, dtype=np.float32)
    if cv2.pointPolygonTest(board_polygon, (probe_x, probe_y), measureDist=False) < 0:
        return None
    return board_point_to_square_name(
        tuple((float(x), float(y)) for x, y in board_polygon.tolist()),
        (probe_x, probe_y),
    )


def _mask_bbox(mask_local: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask_local)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_is_too_large(
    bbox_local: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, ...],
    max_width_fraction: float,
    max_height_fraction: float,
) -> bool:
    x1, y1, x2, y2 = bbox_local
    image_height, image_width = image_shape[:2]
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    return (
        bbox_width > int(round(image_width * max_width_fraction))
        or bbox_height > int(round(image_height * max_height_fraction))
    )


def _get_sam_processor(confidence_threshold: float) -> Any:
    import pipeline.analysis.board_segmenter as board_segmenter

    config = VideoAnalysisConfig(device="cpu", sam_confidence_threshold=confidence_threshold)
    _load_sam(config)
    processor = getattr(board_segmenter, "_sam_processor", None)
    if processor is None:
        raise ValueError("SAM3 processor is unavailable")
    return processor


def _clone_sam_state(state: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(state)
    backbone_out = state.get("backbone_out")
    if isinstance(backbone_out, dict):
        cloned["backbone_out"] = dict(backbone_out)
    return cloned


def _render_original_panel(
    frame: ProposalFrame,
    proposals: list[SquareCropProposal],
) -> np.ndarray:
    panel = frame.image_bgr.copy()
    cv2.polylines(
        panel,
        [np.asarray(frame.corners, dtype=np.int32)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )
    colors = _proposal_colors(len(proposals))
    for color, proposal in zip(colors, proposals, strict=False):
        if proposal.mask is not None:
            panel = _apply_mask_overlay(panel, proposal.mask > 0, color)
        if proposal.bbox is not None:
            x1, y1, x2, y2 = proposal.bbox
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                panel,
                proposal.square,
                (x1 + 4, max(20, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
    return _resize_panel(panel, width=720, title="original image")


def _render_board_panel(
    board_image_bgr: np.ndarray,
    board_x1: int,
    board_y1: int,
    proposals: list[SquareCropProposal],
) -> np.ndarray:
    panel = board_image_bgr.copy()
    colors = _proposal_colors(len(proposals))
    for color, proposal in zip(colors, proposals, strict=False):
        if proposal.mask is not None:
            local_mask = proposal.mask[
                board_y1 : board_y1 + board_image_bgr.shape[0],
                board_x1 : board_x1 + board_image_bgr.shape[1],
            ]
            panel = _apply_mask_overlay(panel, local_mask > 0, color)
        if proposal.bbox is not None:
            x1, y1, x2, y2 = proposal.bbox
            cv2.rectangle(
                panel,
                (x1 - board_x1, y1 - board_y1),
                (x2 - board_x1, y2 - board_y1),
                color,
                2,
            )
            cv2.putText(
                panel,
                proposal.square,
                (x1 - board_x1 + 4, max(20, y1 - board_y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
    return _resize_panel(panel, width=720, title="sam3 masks overlaid")


def _render_crop_grid(
    proposals: list[SquareCropProposal],
    *,
    cell_size: int,
) -> np.ndarray:
    title_height = 36
    if not proposals:
        canvas = np.full((title_height + cell_size, cell_size * 2, 3), 24, dtype=np.uint8)
        cv2.putText(
            canvas,
            "resulting per-square crops",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "no proposals",
            (12, title_height + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return canvas

    columns = min(6, max(1, len(proposals)))
    rows = (len(proposals) + columns - 1) // columns
    canvas = np.full((title_height + rows * cell_size, columns * cell_size, 3), 24, dtype=np.uint8)
    cv2.putText(
        canvas,
        "resulting per-square crops",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    for index, proposal in enumerate(proposals):
        row_index = index // columns
        col_index = index % columns
        top = title_height + row_index * cell_size
        left = col_index * cell_size
        crop = proposal.crop_bgr
        resized = cv2.resize(crop, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
        canvas[top : top + cell_size, left : left + cell_size] = resized
        cv2.putText(
            canvas,
            proposal.square,
            (left + 6, top + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if proposal.score is not None:
            cv2.putText(
                canvas,
                f"{proposal.score:.2f}",
                (left + 6, top + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return canvas


def _apply_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = image_bgr.copy()
    overlay[mask] = (
        (0.75 * overlay[mask].astype(np.float32)) + (0.25 * np.asarray(color, dtype=np.float32))
    ).astype(np.uint8)
    return overlay


def _resize_panel(image_bgr: np.ndarray, *, width: int, title: str) -> np.ndarray:
    height = int(round(image_bgr.shape[0] * (width / image_bgr.shape[1])))
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    header_height = 36
    canvas = np.full((header_height + height, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        canvas,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    canvas[header_height:, :] = resized
    return canvas


def _proposal_colors(count: int) -> list[tuple[int, int, int]]:
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 200, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
    ]
    return [palette[index % len(palette)] for index in range(count)]


__all__ = [
    "Sam3MaskCandidate",
    "Sam3ProposalConfig",
    "propose_sam3",
    "render_sam3_preview",
]
