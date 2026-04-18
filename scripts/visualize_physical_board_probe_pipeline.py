#!/usr/bin/env python3
"""Generate a visual walkthrough of the physical board-probe pipeline.

This is intentionally diagnostic. It shows what one real oblique board frame and
one synthetic training frame become after preprocessing, patch extraction, and
piece-box pooling.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.board_probe.board_data import (
    INPUT_SIZE,
    PhysicalEvalBoardDataset,
    PhysicalSyntheticClipBoardDataset,
    load_annotated_board_frame_bgr,
    prepare_board_neighborhood_geometry,
)
from pipeline.physical.board_probe.probe import (
    _patch_centers,
    board_probe_config_from_checkpoint,
    build_board_state_probe,
    dino_patches_to_square_tokens,
    sample_projected_square_tokens_from_patch_tokens,
)
from pipeline.physical.board_probe.square_probe import load_probe_checkpoint
from pipeline.physical.piece_projection import (
    extract_projected_piece_crop,
    project_piece_bboxes,
    project_square_base_quad,
)
from pipeline.shared import SQUARE_CLASS_NAMES

from argus.model.vision_encoder import VisionEncoder

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / ".agents" / "memory" / "attachments"
_EMPTY_CLASS_ID = 0
_PATCH_GRID_COLOR = (255, 255, 255)
_BOARD_QUAD_COLOR = (80, 220, 100)
_PIECE_BBOX_COLOR = (255, 160, 60)
_CORNER_COLOR = (255, 255, 255)
_OLD_PATCH_COLOR = (80, 160, 255)
_NEW_PATCH_COLOR = (255, 120, 40)


@dataclass(frozen=True)
class SquareExample:
    square_index: int
    square_name: str
    label_name: str
    old_patch_indices: tuple[int, ...]
    new_patch_indices: tuple[int, ...]
    mean_new_patch_sharing: float
    projected_piece_crop_bgr: np.ndarray
    old_top3: list[tuple[str, float]]
    new_top3: list[tuple[str, float]]


@dataclass(frozen=True)
class PoolingStats:
    patches_used: int
    patches_shared: int
    avg_squares_per_used_patch: float
    max_patch_sharing: int
    square_patch_count_min: int
    square_patch_count_mean: float
    square_patch_count_max: int
    adjacent_cosine_old_mean: float
    adjacent_cosine_new_mean: float
    same_square_old_vs_new_cosine_mean: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "patches_used": self.patches_used,
            "patches_shared": self.patches_shared,
            "avg_squares_per_used_patch": self.avg_squares_per_used_patch,
            "max_patch_sharing": self.max_patch_sharing,
            "square_patch_count_min": self.square_patch_count_min,
            "square_patch_count_mean": self.square_patch_count_mean,
            "square_patch_count_max": self.square_patch_count_max,
            "adjacent_cosine_old_mean": self.adjacent_cosine_old_mean,
            "adjacent_cosine_new_mean": self.adjacent_cosine_new_mean,
            "same_square_old_vs_new_cosine_mean": self.same_square_old_vs_new_cosine_mean,
        }


@dataclass(frozen=True)
class SampleAnalysis:
    kind: str
    title: str
    image_rgb: np.ndarray
    scaled_corners: np.ndarray
    labels: tuple[int, ...]
    old_pred_labels: tuple[int, ...]
    new_pred_labels: tuple[int, ...]
    grid_size: int
    patch_centers: np.ndarray
    old_masks: np.ndarray
    new_masks: np.ndarray
    patch_usage_new: np.ndarray
    square_quads: np.ndarray
    piece_bboxes: np.ndarray
    stats: PoolingStats
    occupied_example: SquareExample
    empty_example: SquareExample


@dataclass(frozen=True)
class LoadedModel:
    checkpoint: dict[str, Any]
    encoder: VisionEncoder
    probe: torch.nn.Module


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights_path)
    real_analysis = analyze_real_sample(
        annotation_id=args.annotation_id,
        index=args.real_index,
        model=model,
    )
    synthetic_analysis = analyze_synthetic_sample(index=args.synthetic_index, model=model)

    render_real_walkthrough(real_analysis, output_dir=output_dir)
    render_synthetic_comparison(
        real_analysis,
        synthetic_analysis,
        output_dir=output_dir,
    )
    write_summary(
        real_analysis,
        synthetic_analysis,
        weights_path=args.weights_path,
        output_dir=output_dir,
    )
    print(output_dir / "summary.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a visual walkthrough of the physical board-probe pipeline."
    )
    parser.add_argument("--annotation-id", type=str, default=None)
    parser.add_argument("--real-index", type=int, default=0)
    parser.add_argument("--synthetic-index", type=int, default=0)
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=_PROJECT_ROOT / "weights" / "physical" / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR / "physical_board_probe_pipeline_walkthrough",
    )
    return parser


def load_model(weights_path: Path) -> LoadedModel:
    checkpoint = load_probe_checkpoint(weights_path)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    encoder = VisionEncoder(
        model_name=str(checkpoint["model_name"]),
        frozen=True,
        encoder_type=str(metadata.get("encoder_type", "dinov2")),
        feature_layer_indices=metadata.get("feature_layer_indices"),
    )
    probe = build_board_state_probe(
        encoder.embed_dim,
        probe_config=board_probe_config_from_checkpoint(checkpoint),
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no state_dict: {weights_path}")
    probe.load_state_dict(state_dict)
    probe.eval()
    return LoadedModel(checkpoint=checkpoint, encoder=encoder, probe=probe)


def analyze_real_sample(
    *,
    annotation_id: str | None,
    index: int,
    model: LoadedModel,
) -> SampleAnalysis:
    dataset = PhysicalEvalBoardDataset()
    if annotation_id is None:
        row = dataset.rows[index]
    else:
        row = next(
            candidate for candidate in dataset.rows if candidate.annotation_id == annotation_id
        )
    frame_bgr = load_annotated_board_frame_bgr(row, clip_cache={})
    _board_tensor, scaled_corners, piece_bboxes = prepare_board_neighborhood_geometry(
        frame_bgr,
        row.corners,
        size=INPUT_SIZE,
    )
    resized_crop_rgb = (_board_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    return analyze_sample(
        kind="real",
        title=(
            f"real held-out frame | {row.annotation_id} | source {row.source_video_id or 'unknown'}"
        ),
        image_rgb=resized_crop_rgb,
        scaled_corners=scaled_corners.numpy(),
        labels=row.labels,
        piece_bboxes=piece_bboxes.numpy(),
        model=model,
    )


def analyze_synthetic_sample(*, index: int, model: LoadedModel) -> SampleAnalysis:
    dataset = PhysicalSyntheticClipBoardDataset(num_positions=max(index + 1, 1), seed=42)
    image_tensor, label_tensor, corners_tensor = dataset[index]
    image_rgb = _unnormalize_to_rgb_uint8(image_tensor)
    labels = tuple(int(value) for value in label_tensor.tolist())
    row = dataset.rows[index]
    title = f"synthetic train frame | {Path(row.clip_path).name} | frame {row.frame_index}"
    return analyze_sample(
        kind="synthetic",
        title=title,
        image_rgb=image_rgb,
        scaled_corners=corners_tensor.numpy(),
        labels=labels,
        piece_bboxes=None,
        model=model,
    )


def analyze_sample(
    *,
    kind: str,
    title: str,
    image_rgb: np.ndarray,
    scaled_corners: np.ndarray,
    labels: tuple[int, ...],
    piece_bboxes: np.ndarray | None,
    model: LoadedModel,
) -> SampleAnalysis:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    board_tensor = _rgb_uint8_to_normalized_tensor(image_rgb)

    with torch.no_grad():
        patch_tokens = model.encoder.forward_patches(board_tensor.unsqueeze(0))
        grid_size = int((patch_tokens.shape[1] - 1) ** 0.5)
        old_square_tokens = dino_patches_to_square_tokens(patch_tokens)[0]
        new_square_tokens = sample_projected_square_tokens_from_patch_tokens(
            patch_tokens,
            geometry=(
                torch.from_numpy(scaled_corners).unsqueeze(0)
                if piece_bboxes is None
                else torch.from_numpy(piece_bboxes).unsqueeze(0)
            ),
            image_size=INPUT_SIZE,
        )[0]
        old_logits = model.probe(old_square_tokens.unsqueeze(0))[0]
        new_logits = model.probe(new_square_tokens.unsqueeze(0))[0]

    patch_centers = _patch_centers(
        grid_size=grid_size,
        image_size=INPUT_SIZE,
        device=torch.device("cpu"),
    ).numpy()
    old_masks = build_old_pooling_masks(grid_size=grid_size)
    square_quads = np.stack(
        [
            project_square_base_quad(scaled_corners.tolist(), row=square // 8, col=square % 8)
            for square in range(64)
        ],
        axis=0,
    )
    piece_bboxes = (
        project_piece_bboxes(scaled_corners.tolist(), frame_shape=image_rgb.shape)
        if piece_bboxes is None
        else piece_bboxes
    )
    new_masks = build_piecebox_masks(patch_centers=patch_centers, piece_bboxes=piece_bboxes)
    patch_usage_new = new_masks.sum(axis=0)
    stats = compute_pooling_stats(
        old_square_tokens=old_square_tokens,
        new_square_tokens=new_square_tokens,
        old_masks=old_masks,
        new_masks=new_masks,
        patch_usage_new=patch_usage_new,
    )
    occupied_example, empty_example = choose_examples(
        image_bgr=image_bgr,
        corners=scaled_corners,
        labels=labels,
        old_masks=old_masks,
        new_masks=new_masks,
        patch_usage_new=patch_usage_new,
        old_logits=old_logits,
        new_logits=new_logits,
    )
    return SampleAnalysis(
        kind=kind,
        title=title,
        image_rgb=image_rgb,
        scaled_corners=scaled_corners,
        labels=labels,
        old_pred_labels=tuple(int(value) for value in old_logits.argmax(dim=1).tolist()),
        new_pred_labels=tuple(int(value) for value in new_logits.argmax(dim=1).tolist()),
        grid_size=grid_size,
        patch_centers=patch_centers,
        old_masks=old_masks,
        new_masks=new_masks,
        patch_usage_new=patch_usage_new,
        square_quads=square_quads,
        piece_bboxes=piece_bboxes,
        stats=stats,
        occupied_example=occupied_example,
        empty_example=empty_example,
    )


def render_real_walkthrough(analysis: SampleAnalysis, *, output_dir: Path) -> None:
    frame_path = output_dir / "01_real_input.png"
    overlay_path = output_dir / "02_real_geometry_overlay.png"
    heatmap_path = output_dir / "03_real_patch_usage_heatmap.png"
    examples_path = output_dir / "04_real_square_examples.png"
    explanation_path = output_dir / "05_real_probe_boards.png"

    _save_rgb(frame_path, render_input_image(analysis))
    _save_rgb(overlay_path, render_geometry_overlay(analysis))
    _save_rgb(heatmap_path, render_patch_usage_heatmap(analysis))
    _save_rgb(examples_path, render_square_examples(analysis))
    _save_rgb(explanation_path, render_probe_board_panel(analysis))


def render_synthetic_comparison(
    real_analysis: SampleAnalysis,
    synthetic_analysis: SampleAnalysis,
    *,
    output_dir: Path,
) -> None:
    comparison_path = output_dir / "06_real_vs_synthetic_pooling.png"
    _save_rgb(
        comparison_path,
        render_real_vs_synthetic_panel(real_analysis, synthetic_analysis),
    )


def write_summary(
    real_analysis: SampleAnalysis,
    synthetic_analysis: SampleAnalysis,
    *,
    weights_path: Path,
    output_dir: Path,
) -> None:
    summary_path = output_dir / "summary.md"
    stats_path = output_dir / "stats.json"

    stats_payload = {
        "weights_path": str(weights_path.relative_to(_PROJECT_ROOT)),
        "real": {
            "title": real_analysis.title,
            "stats": real_analysis.stats.to_dict(),
            "occupied_example": square_example_to_dict(real_analysis.occupied_example),
            "empty_example": square_example_to_dict(real_analysis.empty_example),
        },
        "synthetic": {
            "title": synthetic_analysis.title,
            "stats": synthetic_analysis.stats.to_dict(),
            "occupied_example": square_example_to_dict(synthetic_analysis.occupied_example),
            "empty_example": square_example_to_dict(synthetic_analysis.empty_example),
        },
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True))

    real_stats = real_analysis.stats
    synthetic_stats = synthetic_analysis.stats
    real_occ = real_analysis.occupied_example
    real_empty = real_analysis.empty_example

    lines = [
        "# Physical board-probe pipeline walkthrough",
        "",
        f"Weights: `{weights_path.relative_to(_PROJECT_ROOT)}`",
        "",
        "## What the model actually sees",
        "",
        "It does **not** crop 64 piece images and classify them one by one.",
        "",
        "For one 224×224 board input:",
        "",
        "1. run frozen DINO once over the whole board",
        f"2. get a `{real_analysis.grid_size}x{real_analysis.grid_size}` patch grid "
        f"(`{real_analysis.grid_size * real_analysis.grid_size}` patches + 1 CLS token)`",
        "3. build one 64-token board representation by averaging patch embeddings per square",
        "4. feed those 64 tokens into the board probe head",
        "",
        "The key change is **step 3**.",
        "",
        "- Old planar pooling: each square = its own disjoint 2×2 patch block",
        "- New piece-box pooling: each square = all patches whose centers fall inside the "
        "projected 3D piece bbox for that square",
        "",
        "So this is **not** just ‘same architecture with better chess-piece crops’. "
        "It is a different tokenization of the board.",
        "",
        "## Files",
        "",
        (
            "- `01_real_input.png` — resized real board input with corners, patch grid, "
            "and 8×8 labels"
        ),
        (
            "- `02_real_geometry_overlay.png` — base-board quads (green) vs projected "
            "piece boxes (orange)"
        ),
        "- `03_real_patch_usage_heatmap.png` — how many different squares reuse each patch",
        ("- `04_real_square_examples.png` — one occupied square and one empty square, old vs new"),
        (
            "- `05_real_probe_boards.png` — old planar tokens vs new piece-box tokens "
            "on the same frame"
        ),
        ("- `06_real_vs_synthetic_pooling.png` — real oblique frame vs synthetic train frame"),
        "- `stats.json` — raw numbers",
        "",
        "## Real frame numbers",
        "",
        f"Frame: `{real_analysis.title}`",
        "",
        "- old planar patches per square: always `4`",
        f"- new piece-box patches per square: `min {real_stats.square_patch_count_min}`, "
        f"`mean {real_stats.square_patch_count_mean:.2f}`, "
        f"`max {real_stats.square_patch_count_max}`",
        f"- patches touched by any square: `{real_stats.patches_used}/256`",
        f"- patches reused by more than one square: `{real_stats.patches_shared}/256`",
        f"- average squares per used patch: `{real_stats.avg_squares_per_used_patch:.2f}`",
        f"- maximum sharing of one patch: `{real_stats.max_patch_sharing}` squares",
        f"- adjacent-token cosine, old planar: `{real_stats.adjacent_cosine_old_mean:.3f}`",
        f"- adjacent-token cosine, new piece-box: `{real_stats.adjacent_cosine_new_mean:.3f}`",
        f"- same square old-vs-new token cosine: "
        f"`{real_stats.same_square_old_vs_new_cosine_mean:.3f}`",
        "",
        "Interpretation: on a real oblique board, neighboring square tokens become much more "
        "similar after piece-box pooling because the projected boxes overlap heavily.",
        "",
        "## Two concrete square examples from the real frame",
        "",
        f"Occupied example: `{real_occ.square_name}` (`{real_occ.label_name}`)",
        f"- old planar patch count: `{len(real_occ.old_patch_indices)}`",
        f"- new piece-box patch count: `{len(real_occ.new_patch_indices)}`",
        (
            "- mean patch sharing inside new token: "
            f"`{real_occ.mean_new_patch_sharing:.2f}` squares/patch"
        ),
        f"- old top-3 classes: `{format_top3(real_occ.old_top3)}`",
        f"- new top-3 classes: `{format_top3(real_occ.new_top3)}`",
        "",
        f"Empty example: `{real_empty.square_name}` (`{real_empty.label_name}`)",
        f"- old planar patch count: `{len(real_empty.old_patch_indices)}`",
        f"- new piece-box patch count: `{len(real_empty.new_patch_indices)}`",
        (
            "- mean patch sharing inside new token: "
            f"`{real_empty.mean_new_patch_sharing:.2f}` squares/patch"
        ),
        f"- old top-3 classes: `{format_top3(real_empty.old_top3)}`",
        f"- new top-3 classes: `{format_top3(real_empty.new_top3)}`",
        "",
        "Notice the empty square still gets a full hypothetical **piece** region under the new "
        "pooling rule. The board probe must learn ‘empty’ from that projected piece-shaped region, "
        "not from the flat square itself.",
        "",
        "## Why training can break even if the geometry looks more semantically correct",
        "",
        f"Synthetic sample: `{synthetic_analysis.title}`",
        "",
        (
            "- synthetic mean piece-box patches per square: "
            f"`{synthetic_stats.square_patch_count_mean:.2f}`"
        ),
        (
            "- synthetic average squares per used patch: "
            f"`{synthetic_stats.avg_squares_per_used_patch:.2f}`"
        ),
        f"- synthetic max patch sharing: `{synthetic_stats.max_patch_sharing}`",
        f"- real average squares per used patch: `{real_stats.avg_squares_per_used_patch:.2f}`",
        f"- real max patch sharing: `{real_stats.max_patch_sharing}`",
        "",
        (
            "That means the synthetic train distribution is much less overlapped than "
            "the real oblique distribution. The training token geometry and the real "
            "token geometry are no longer the same."
        ),
        "",
        "## Most likely failure modes to inspect next",
        "",
        (
            "1. **Patch overlap / token smearing** — a single DINO patch can contribute "
            "to many squares."
        ),
        (
            "2. **Empty-square mismatch** — empty squares now pool a piece-shaped "
            "region, not the board surface."
        ),
        (
            "3. **Synthetic-vs-real geometry gap** — synthetic boards are closer to "
            "top-down; real boards are much more oblique and overlapping."
        ),
        (
            "4. **Selection mismatch** — if model selection is driven by synthetic-style "
            "validation, it may pick checkpoints that do not help on real oblique "
            "piece-box tokens."
        ),
        "",
        (
            "If you want, the next useful step is to extend this into a multi-frame "
            "failure viewer that shows these same overlays on a bad real clip while "
            "plotting per-square logits over time."
        ),
    ]
    summary_path.write_text("\n".join(lines) + "\n")


def square_example_to_dict(example: SquareExample) -> dict[str, Any]:
    return {
        "square_name": example.square_name,
        "label_name": example.label_name,
        "old_patch_indices": list(example.old_patch_indices),
        "new_patch_indices": list(example.new_patch_indices),
        "mean_new_patch_sharing": example.mean_new_patch_sharing,
        "old_top3": example.old_top3,
        "new_top3": example.new_top3,
    }


def render_input_image(analysis: SampleAnalysis) -> np.ndarray:
    image = analysis.image_rgb.copy()
    _draw_patch_grid(image, analysis.grid_size, color=_PATCH_GRID_COLOR, alpha=0.35)
    _draw_square_labels(image)
    _draw_corners(image, analysis.scaled_corners)
    return image


def render_geometry_overlay(analysis: SampleAnalysis) -> np.ndarray:
    image = analysis.image_rgb.copy()
    overlay = image.copy()
    for square in range(64):
        quad = analysis.square_quads[square].astype(np.int32)
        bbox = analysis.piece_bboxes[square]
        cv2.polylines(overlay, [quad], isClosed=True, color=_BOARD_QUAD_COLOR, thickness=1)
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _PIECE_BBOX_COLOR, 1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0.0, dst=image)
    _draw_patch_grid(image, analysis.grid_size, color=_PATCH_GRID_COLOR, alpha=0.15)
    _draw_corners(image, analysis.scaled_corners)
    return image


def render_patch_usage_heatmap(analysis: SampleAnalysis) -> np.ndarray:
    image = analysis.image_rgb.copy()
    usage_grid = analysis.patch_usage_new.reshape(analysis.grid_size, analysis.grid_size)
    normalized = np.clip(
        usage_grid.astype(np.float32) / max(float(usage_grid.max()), 1.0),
        0.0,
        1.0,
    )
    heatmap_small = (normalized * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(
        cv2.resize(heatmap_small, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_TURBO,
    )
    blended = cv2.addWeighted(image, 0.45, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.55, 0.0)
    _draw_patch_grid(blended, analysis.grid_size, color=_PATCH_GRID_COLOR, alpha=0.25)
    return _annotate_image(
        blended,
        [
            f"max sharing: {analysis.stats.max_patch_sharing} squares/patch",
            f"mean sharing: {analysis.stats.avg_squares_per_used_patch:.2f}",
            f"shared patches: {analysis.stats.patches_shared}/256",
        ],
    )


def render_square_examples(analysis: SampleAnalysis) -> np.ndarray:
    occupied_row = render_square_example_row(
        analysis,
        example=analysis.occupied_example,
        heading="occupied square example",
    )
    empty_row = render_square_example_row(
        analysis,
        example=analysis.empty_example,
        heading="empty square example",
    )
    return _stack_vertical([occupied_row, empty_row], gap=16, bg=(20, 20, 20))


def render_square_example_row(
    analysis: SampleAnalysis,
    *,
    example: SquareExample,
    heading: str,
) -> np.ndarray:
    old_panel = render_patch_mask_panel(
        analysis,
        patch_indices=example.old_patch_indices,
        square_index=example.square_index,
        color=_OLD_PATCH_COLOR,
        title=(
            f"old planar pooling | {example.square_name} | {len(example.old_patch_indices)} patches"
        ),
    )
    new_panel = render_patch_mask_panel(
        analysis,
        patch_indices=example.new_patch_indices,
        square_index=example.square_index,
        color=_NEW_PATCH_COLOR,
        title=(
            f"new piece-box pooling | {example.square_name} | "
            f"{len(example.new_patch_indices)} patches | "
            f"mean sharing {example.mean_new_patch_sharing:.2f}"
        ),
    )
    crop_panel = render_projected_piece_crop_panel(
        example,
        title=(
            f"intuitive piece crop only | old top3 {format_top3(example.old_top3)} | "
            f"new top3 {format_top3(example.new_top3)}"
        ),
    )
    row = _stack_horizontal([old_panel, new_panel, crop_panel], gap=12, bg=(20, 20, 20))
    return _annotate_image(
        row,
        [f"{heading}: {example.square_name} = {example.label_name}"],
        font_size=22,
        top_padding=42,
    )


def render_patch_mask_panel(
    analysis: SampleAnalysis,
    *,
    patch_indices: tuple[int, ...],
    square_index: int,
    color: tuple[int, int, int],
    title: str,
) -> np.ndarray:
    image = analysis.image_rgb.copy()
    overlay = image.copy()
    for patch_index in patch_indices:
        row = patch_index // analysis.grid_size
        col = patch_index % analysis.grid_size
        x1 = int(round(col * INPUT_SIZE / analysis.grid_size))
        y1 = int(round(row * INPUT_SIZE / analysis.grid_size))
        x2 = int(round((col + 1) * INPUT_SIZE / analysis.grid_size))
        y2 = int(round((row + 1) * INPUT_SIZE / analysis.grid_size))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)
    quad = analysis.square_quads[square_index].astype(np.int32)
    x1, y1, x2, y2 = (int(round(value)) for value in analysis.piece_bboxes[square_index])
    cv2.polylines(image, [quad], isClosed=True, color=_BOARD_QUAD_COLOR, thickness=2)
    cv2.rectangle(image, (x1, y1), (x2, y2), _PIECE_BBOX_COLOR, 2)
    _draw_patch_grid(image, analysis.grid_size, color=_PATCH_GRID_COLOR, alpha=0.3)
    return _annotate_image(image, [title])


def render_projected_piece_crop_panel(example: SquareExample, *, title: str) -> np.ndarray:
    crop_rgb = cv2.cvtColor(example.projected_piece_crop_bgr, cv2.COLOR_BGR2RGB)
    return _annotate_image(crop_rgb, [title])


def render_probe_board_panel(analysis: SampleAnalysis) -> np.ndarray:
    gt_labels = analysis.labels
    old_labels = analysis.old_pred_labels
    new_labels = analysis.new_pred_labels
    panels = [
        render_text_board(gt_labels, title="ground truth"),
        render_text_board(
            old_labels,
            title=(
                f"same probe, old planar tokens | errors "
                f"{sum(int(a != b) for a, b in zip(gt_labels, old_labels))}"
            ),
            target_labels=gt_labels,
        ),
        render_text_board(
            new_labels,
            title=(
                f"same probe, new piece-box tokens | errors "
                f"{sum(int(a != b) for a, b in zip(gt_labels, new_labels))}"
            ),
            target_labels=gt_labels,
        ),
    ]
    return _stack_horizontal(panels, gap=12, bg=(20, 20, 20))


def render_real_vs_synthetic_panel(
    real_analysis: SampleAnalysis,
    synthetic_analysis: SampleAnalysis,
) -> np.ndarray:
    top = _stack_horizontal(
        [
            _annotate_image(
                render_geometry_overlay(real_analysis),
                [real_analysis.title],
            ),
            _annotate_image(
                render_geometry_overlay(synthetic_analysis),
                [synthetic_analysis.title],
            ),
        ],
        gap=12,
        bg=(20, 20, 20),
    )
    bottom = _stack_horizontal(
        [
            _annotate_image(
                render_patch_usage_heatmap(real_analysis),
                [
                    f"real mean patches/square {real_analysis.stats.square_patch_count_mean:.2f}",
                    f"real mean share {real_analysis.stats.avg_squares_per_used_patch:.2f}",
                ],
                top_padding=42,
            ),
            _annotate_image(
                render_patch_usage_heatmap(synthetic_analysis),
                [
                    f"synthetic mean patches/square "
                    f"{synthetic_analysis.stats.square_patch_count_mean:.2f}",
                    f"synthetic mean share "
                    f"{synthetic_analysis.stats.avg_squares_per_used_patch:.2f}",
                ],
                top_padding=42,
            ),
        ],
        gap=12,
        bg=(20, 20, 20),
    )
    return _stack_vertical([top, bottom], gap=16, bg=(20, 20, 20))


def choose_examples(
    *,
    image_bgr: np.ndarray,
    corners: np.ndarray,
    labels: tuple[int, ...],
    old_masks: np.ndarray,
    new_masks: np.ndarray,
    patch_usage_new: np.ndarray,
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> tuple[SquareExample, SquareExample]:
    occupied_indices = [index for index, label in enumerate(labels) if label != _EMPTY_CLASS_ID]
    empty_indices = [index for index, label in enumerate(labels) if label == _EMPTY_CLASS_ID]
    occupied_index = max(
        occupied_indices,
        key=lambda index: _square_overlap_score(index, new_masks, patch_usage_new),
    )
    empty_index = max(
        empty_indices,
        key=lambda index: _square_overlap_score(index, new_masks, patch_usage_new),
    )
    return (
        build_square_example(
            square_index=occupied_index,
            image_bgr=image_bgr,
            corners=corners,
            labels=labels,
            old_masks=old_masks,
            new_masks=new_masks,
            patch_usage_new=patch_usage_new,
            old_logits=old_logits,
            new_logits=new_logits,
        ),
        build_square_example(
            square_index=empty_index,
            image_bgr=image_bgr,
            corners=corners,
            labels=labels,
            old_masks=old_masks,
            new_masks=new_masks,
            patch_usage_new=patch_usage_new,
            old_logits=old_logits,
            new_logits=new_logits,
        ),
    )


def build_square_example(
    *,
    square_index: int,
    image_bgr: np.ndarray,
    corners: np.ndarray,
    labels: tuple[int, ...],
    old_masks: np.ndarray,
    new_masks: np.ndarray,
    patch_usage_new: np.ndarray,
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> SquareExample:
    row = square_index // 8
    col = square_index % 8
    square_name = f"{chr(ord('a') + col)}{8 - row}"
    label_name = SQUARE_CLASS_NAMES[labels[square_index]]
    new_patch_indices = tuple(np.flatnonzero(new_masks[square_index]).tolist())
    old_patch_indices = tuple(np.flatnonzero(old_masks[square_index]).tolist())
    mean_new_patch_sharing = float(patch_usage_new[list(new_patch_indices)].mean())
    crop = extract_projected_piece_crop(
        image_bgr,
        corners.tolist(),
        row=row,
        col=col,
        output_size=INPUT_SIZE,
    )
    return SquareExample(
        square_index=square_index,
        square_name=square_name,
        label_name=label_name,
        old_patch_indices=old_patch_indices,
        new_patch_indices=new_patch_indices,
        mean_new_patch_sharing=mean_new_patch_sharing,
        projected_piece_crop_bgr=crop,
        old_top3=top3_from_logits(old_logits[square_index]),
        new_top3=top3_from_logits(new_logits[square_index]),
    )


def _square_overlap_score(
    square_index: int,
    new_masks: np.ndarray,
    patch_usage_new: np.ndarray,
) -> float:
    patch_indices = np.flatnonzero(new_masks[square_index])
    return float(patch_usage_new[patch_indices].mean())


def build_old_pooling_masks(*, grid_size: int) -> np.ndarray:
    patches_per_square = grid_size // 8
    masks = np.zeros((64, grid_size * grid_size), dtype=bool)
    for row in range(8):
        for col in range(8):
            square_index = row * 8 + col
            for patch_row in range(row * patches_per_square, (row + 1) * patches_per_square):
                for patch_col in range(col * patches_per_square, (col + 1) * patches_per_square):
                    patch_index = patch_row * grid_size + patch_col
                    masks[square_index, patch_index] = True
    return masks


def build_piecebox_masks(
    *,
    patch_centers: np.ndarray,
    piece_bboxes: np.ndarray,
) -> np.ndarray:
    masks = np.zeros((64, patch_centers.shape[0]), dtype=bool)
    for square_index, bbox in enumerate(piece_bboxes):
        xmin, ymin, xmax, ymax = bbox
        mask = (
            (patch_centers[:, 0] >= xmin)
            & (patch_centers[:, 0] <= xmax)
            & (patch_centers[:, 1] >= ymin)
            & (patch_centers[:, 1] <= ymax)
        )
        if not mask.any():
            distances = np.sum(
                (patch_centers - np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])) ** 2,
                axis=1,
            )
            mask[int(distances.argmin())] = True
        masks[square_index] = mask
    return masks


def compute_pooling_stats(
    *,
    old_square_tokens: torch.Tensor,
    new_square_tokens: torch.Tensor,
    old_masks: np.ndarray,
    new_masks: np.ndarray,
    patch_usage_new: np.ndarray,
) -> PoolingStats:
    square_patch_counts = new_masks.sum(axis=1)
    adjacent_old: list[float] = []
    adjacent_new: list[float] = []
    same_old_vs_new: list[float] = []
    for row in range(8):
        for col in range(8):
            square_index = row * 8 + col
            same_old_vs_new.append(
                float(
                    torch.nn.functional.cosine_similarity(
                        old_square_tokens[square_index],
                        new_square_tokens[square_index],
                        dim=0,
                    ).item()
                )
            )
            if col < 7:
                right = square_index + 1
                adjacent_old.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            old_square_tokens[square_index],
                            old_square_tokens[right],
                            dim=0,
                        ).item()
                    )
                )
                adjacent_new.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            new_square_tokens[square_index],
                            new_square_tokens[right],
                            dim=0,
                        ).item()
                    )
                )
            if row < 7:
                down = square_index + 8
                adjacent_old.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            old_square_tokens[square_index],
                            old_square_tokens[down],
                            dim=0,
                        ).item()
                    )
                )
                adjacent_new.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            new_square_tokens[square_index],
                            new_square_tokens[down],
                            dim=0,
                        ).item()
                    )
                )
    used_patch_mask = patch_usage_new > 0
    return PoolingStats(
        patches_used=int(used_patch_mask.sum()),
        patches_shared=int((patch_usage_new > 1).sum()),
        avg_squares_per_used_patch=float(patch_usage_new[used_patch_mask].mean()),
        max_patch_sharing=int(patch_usage_new.max()),
        square_patch_count_min=int(square_patch_counts.min()),
        square_patch_count_mean=float(square_patch_counts.mean()),
        square_patch_count_max=int(square_patch_counts.max()),
        adjacent_cosine_old_mean=float(np.mean(adjacent_old)),
        adjacent_cosine_new_mean=float(np.mean(adjacent_new)),
        same_square_old_vs_new_cosine_mean=float(np.mean(same_old_vs_new)),
    )


def top3_from_logits(logits: torch.Tensor) -> list[tuple[str, float]]:
    probabilities = torch.softmax(logits, dim=0)
    values, indices = probabilities.topk(3)
    return [
        (SQUARE_CLASS_NAMES[int(index)], float(value))
        for value, index in zip(values.tolist(), indices.tolist())
    ]


def render_text_board(
    labels: tuple[int, ...],
    *,
    title: str,
    target_labels: tuple[int, ...] | None = None,
) -> np.ndarray:
    cell = 34
    title_height = 36
    size = cell * 8
    image = Image.new("RGB", (size, size + title_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = _load_font(18)
    small_font = _load_font(14)
    draw.text((8, 8), title, fill=(20, 20, 20), font=small_font)
    for row in range(8):
        for col in range(8):
            x1 = col * cell
            y1 = title_height + row * cell
            x2 = x1 + cell
            y2 = y1 + cell
            fill = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
            draw.rectangle((x1, y1, x2, y2), fill=fill)
            square_index = row * 8 + col
            symbol = SQUARE_CLASS_NAMES[labels[square_index]]
            text = "" if symbol == "empty" else symbol
            if target_labels is not None and labels[square_index] != target_labels[square_index]:
                draw.rectangle((x1 + 1, y1 + 1, x2 - 1, y2 - 1), outline=(220, 70, 70), width=3)
            if text:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.text(
                    (x1 + (cell - text_w) / 2, y1 + (cell - text_h) / 2 - 1),
                    text,
                    fill=(20, 20, 20),
                    font=font,
                )
    return np.asarray(image)


def format_top3(top3: list[tuple[str, float]]) -> str:
    return ", ".join(f"{name}:{score:.2f}" for name, score in top3)


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def _annotate_image(
    image_rgb: np.ndarray,
    lines: list[str],
    *,
    font_size: int = 16,
    top_padding: int = 28,
) -> np.ndarray:
    font = _load_font(font_size)
    width = image_rgb.shape[1]
    text_canvas = Image.new("RGB", (width, top_padding), (24, 24, 24))
    draw = ImageDraw.Draw(text_canvas)
    y = 4
    for line in lines:
        draw.text((8, y), line, fill=(245, 245, 245), font=font)
        y += font_size + 2
    return _stack_vertical([np.asarray(text_canvas), image_rgb], gap=0, bg=(24, 24, 24))


def _stack_horizontal(
    images: list[np.ndarray],
    *,
    gap: int,
    bg: tuple[int, int, int],
) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    x = 0
    for image in images:
        y = (height - image.shape[0]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return canvas


def _stack_vertical(
    images: list[np.ndarray],
    *,
    gap: int,
    bg: tuple[int, int, int],
) -> np.ndarray:
    width = max(image.shape[1] for image in images)
    height = sum(image.shape[0] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    y = 0
    for image in images:
        x = (width - image.shape[1]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        y += image.shape[0] + gap
    return canvas


def _draw_patch_grid(
    image_rgb: np.ndarray,
    grid_size: int,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image_rgb.copy()
    for step in range(1, grid_size):
        x = int(round(step * INPUT_SIZE / grid_size))
        y = int(round(step * INPUT_SIZE / grid_size))
        cv2.line(overlay, (x, 0), (x, INPUT_SIZE), color, 1)
        cv2.line(overlay, (0, y), (INPUT_SIZE, y), color, 1)
    cv2.addWeighted(overlay, alpha, image_rgb, 1.0 - alpha, 0.0, dst=image_rgb)


def _draw_square_labels(image_rgb: np.ndarray) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in range(8):
        for col in range(8):
            square_name = f"{chr(ord('a') + col)}{8 - row}"
            x = int(round((col + 0.1) * INPUT_SIZE / 8.0))
            y = int(round((row + 0.25) * INPUT_SIZE / 8.0))
            cv2.putText(image_rgb, square_name, (x, y), font, 0.32, (255, 80, 80), 1, cv2.LINE_AA)


def _draw_corners(image_rgb: np.ndarray, corners: np.ndarray) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, point in enumerate(corners):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(image_rgb, (x, y), 4, _CORNER_COLOR, -1)
        cv2.putText(
            image_rgb,
            f"c{index}",
            (x + 6, y - 4),
            font,
            0.4,
            _CORNER_COLOR,
            1,
            cv2.LINE_AA,
        )


def _resized_crop_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    )


def _unnormalize_to_rgb_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    rgb = (image_tensor * std + mean).clamp(0.0, 1.0)
    return (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _rgb_uint8_to_normalized_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _full_frame_corners(image_bgr: np.ndarray) -> tuple[tuple[float, float], ...]:
    height, width = image_bgr.shape[:2]
    return (
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    )


if __name__ == "__main__":
    main()
