"""Render five Stage-1 visibility debug overlays from `study/eval/labels.jsonl`.

Selection picks one frame near each target piece count (30, 26, 22, 18, 12) —
denser frames first — and prefers a distinct `source_video_id` per pick so the
five overlays span multiple camera setups. Output goes to
`study/templates-v2/geometry/debug_stage1_v2/` to avoid clashing with the
existing `debug/` and `debug-stage1/` directories from the failed attempt.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
for _path in (_PROJECT_ROOT, _THIS_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from collections.abc import Iterable, Mapping
from typing import Any

import cv2
import numpy as np

from pipeline.physical.piece_projection import (
    extract_board_neighborhood_crop,
    project_square_base_quad,
)

from visibility import (
    FrameVisibility,
    PieceVisibility,
    board_fen_from_piece_entries,
    compute_frame_visibility,
)

PROJECT_ROOT = _PROJECT_ROOT
DEFAULT_LABELS_PATH = PROJECT_ROOT / "study" / "eval" / "labels.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "study" / "templates-v2" / "geometry" / "debug_stage1_v2"
TARGET_PIECE_COUNTS: tuple[int, ...] = (30, 26, 22, 18, 12)

# Videos whose eval-label corners systematically overshoot the playing surface.
# Refining them requires a separate corner-refinement pass (spawned task). Until
# that lands, skip these videos to avoid grids that don't align with the board.
EXCLUDED_VIDEO_IDS: frozenset[str] = frozenset({"9h4IE1G99OE"})

VISIBLE_COLOR_BGR = np.array((72, 196, 96), dtype=np.float32)
OCCLUDED_COLOR_BGR = np.array((40, 52, 224), dtype=np.float32)
SILHOUETTE_OUTLINE_BGR = (255, 255, 255)
VISIBLE_ALPHA = 0.32
OCCLUDED_ALPHA = 0.48
LABEL_VISIBILITY_THRESHOLD = 0.999


def load_labels(labels_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in labels_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _has_required_fields(row: Mapping[str, Any]) -> bool:
    image_path = row.get("image_path")
    pieces = row.get("pieces")
    corners = row.get("corners")
    frame_id = row.get("frame_id")
    corner_space = row.get("corner_space")
    source_video_id = row.get("source_video_id")
    if not isinstance(image_path, str) or not isinstance(frame_id, str):
        return False
    if not isinstance(pieces, list) or not isinstance(corners, list):
        return False
    if len(corners) != 4:
        return False
    if corner_space != "native_frame":
        return False
    if source_video_id in EXCLUDED_VIDEO_IDS:
        return False
    return True


def _count_pieces(row: Mapping[str, Any]) -> int:
    pieces = row.get("pieces")
    if not isinstance(pieces, list):
        return 0
    return sum(
        1 for piece in pieces if isinstance(piece, Mapping) and isinstance(piece.get("square"), str)
    )


SAME_VIDEO_PENALTY = 2


def select_debug_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    targets: tuple[int, ...] = TARGET_PIECE_COUNTS,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not _has_required_fields(row):
            continue
        enriched = dict(row)
        enriched["piece_count"] = _count_pieces(row)
        candidates.append(enriched)

    selected: list[dict[str, Any]] = []
    used_frame_ids: set[str] = set()
    used_videos: set[str] = set()
    for target in targets:
        pool = [row for row in candidates if row["frame_id"] not in used_frame_ids]
        if not pool:
            break
        pool.sort(
            key=lambda row: (
                abs(int(row["piece_count"]) - target)
                + (SAME_VIDEO_PENALTY if row.get("source_video_id") in used_videos else 0),
                str(row.get("category", "")),
                str(row["frame_id"]),
            )
        )
        choice = pool[0]
        selected.append(choice)
        used_frame_ids.add(choice["frame_id"])
        source_video_id = choice.get("source_video_id")
        if isinstance(source_video_id, str):
            used_videos.add(source_video_id)
    return selected


def _alpha_blend(
    canvas: np.ndarray,
    mask: np.ndarray,
    color_bgr: np.ndarray,
    alpha: float,
) -> np.ndarray:
    boolean_mask = mask.astype(bool)
    if not boolean_mask.any():
        return canvas
    canvas[boolean_mask] = (1.0 - alpha) * canvas[boolean_mask] + alpha * color_bgr
    return canvas


def _cuboid_top_anchor(piece: PieceVisibility) -> tuple[int, int]:
    vertices = piece.cuboid_vertices
    top = vertices[4:]
    anchor = top.mean(axis=0)
    return int(round(float(anchor[0]))), int(round(float(anchor[1])))


def _draw_label(
    canvas: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    *,
    font_scale: float = 0.42,
) -> None:
    x, y = anchor
    y = int(np.clip(y, 12, canvas.shape[0] - 4))
    x = int(np.clip(x, 2, canvas.shape[1] - 4))
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_board_grid(
    canvas: np.ndarray,
    corners: Sequence[Sequence[float]],
    *,
    color_bgr: tuple[int, int, int] = (255, 200, 0),
    thickness: int = 1,
) -> None:
    for row in range(8):
        for col in range(8):
            quad = project_square_base_quad(corners, row=row, col=col)
            polygon = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas, [polygon], isClosed=True, color=color_bgr, thickness=thickness
            )


def render_debug_overlay(
    image_bgr: np.ndarray,
    frame: FrameVisibility,
    corners: Sequence[Sequence[float]],
    *,
    title: str,
) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")
    if image_bgr.shape[:2] != frame.image_shape:
        raise ValueError(
            f"image_bgr shape {image_bgr.shape[:2]} does not match frame.image_shape "
            f"{frame.image_shape}"
        )

    canvas = image_bgr.astype(np.float32)
    for piece in frame.pieces.values():
        if piece.own_mask_full is None or piece.visible_mask_full is None:
            raise ValueError("render_debug_overlay requires include_full_masks=True")
        visible = piece.visible_mask_full.astype(bool)
        occluded = piece.own_mask_full.astype(bool) & ~visible
        canvas = _alpha_blend(canvas, occluded, OCCLUDED_COLOR_BGR, OCCLUDED_ALPHA)
        canvas = _alpha_blend(canvas, visible, VISIBLE_COLOR_BGR, VISIBLE_ALPHA)

    canvas = np.clip(canvas, 0.0, 255.0).astype(np.uint8)

    _draw_board_grid(canvas, corners)

    for piece in frame.pieces.values():
        polygon = np.round(piece.silhouette).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], isClosed=True, color=SILHOUETTE_OUTLINE_BGR, thickness=1)

    for piece in frame.pieces.values():
        if piece.visibility_fraction >= LABEL_VISIBILITY_THRESHOLD:
            continue
        label = f"{piece.square} {piece.symbol} {piece.visibility_fraction:.2f}"
        _draw_label(canvas, label, _cuboid_top_anchor(piece))

    _draw_label(canvas, title, (14, 28), font_scale=0.7)
    _draw_label(
        canvas,
        "green=visible  red=occluded-by-closer-cuboid  label hidden when vis=1.00",
        (14, 56),
        font_scale=0.5,
    )
    return canvas


def _visibility_summary(frame: FrameVisibility) -> dict[str, Any]:
    visibility_values = [piece.visibility_fraction for piece in frame.pieces.values()]
    if not visibility_values:
        return {"count": 0, "min": None, "max": None, "mean": None, "fully_visible_count": 0}
    return {
        "count": len(visibility_values),
        "min": float(np.min(visibility_values)),
        "max": float(np.max(visibility_values)),
        "mean": float(np.mean(visibility_values)),
        "fully_visible_count": sum(
            1 for value in visibility_values if value >= LABEL_VISIBILITY_THRESHOLD
        ),
    }


def build_stage1_debug_artifacts(
    *,
    labels_path: Path = DEFAULT_LABELS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    targets: tuple[int, ...] = TARGET_PIECE_COUNTS,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_labels(labels_path)
    selected = select_debug_rows(rows, targets=targets)
    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        image_path = row["image_path"]
        frame_id = row["frame_id"]
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read debug image: {image_path}")
        fen = board_fen_from_piece_entries(row["pieces"])
        frame_visibility = compute_frame_visibility(
            image_shape=image_bgr.shape[:2],
            corners=row["corners"],
            fen=fen,
            include_full_masks=True,
        )
        title = (
            f"{index:02d} {frame_id} cat={row.get('category')} pieces={row['piece_count']}"
        )
        overlay = render_debug_overlay(image_bgr, frame_visibility, row["corners"], title=title)
        output_path = output_dir / f"{index:02d}_{frame_id}.png"
        cv2.imwrite(str(output_path), overlay)

        neighborhood = extract_board_neighborhood_crop(overlay, row["corners"], crop_margin=0.22)
        crop_path = output_dir / f"{index:02d}_{frame_id}_board.png"
        cv2.imwrite(str(crop_path), neighborhood.image_bgr)

        manifest.append(
            {
                "rank": index,
                "frame_id": frame_id,
                "category": row.get("category"),
                "source_video_id": row.get("source_video_id"),
                "piece_count": int(row["piece_count"]),
                "target_piece_count": int(targets[index - 1]),
                "board_fen": fen,
                "image_path": image_path,
                "output_path": str(output_path),
                "board_crop_path": str(crop_path),
                "patch_grid_shape": list(frame_visibility.patch_grid_shape),
                "occlusion_order": list(frame_visibility.occlusion_order),
                "visibility": _visibility_summary(frame_visibility),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_stage1_debug_artifacts(
        labels_path=Path(args.labels_path),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
