"""Render a Stage-2 contact sheet: 20 proposal crops × 3 masks.

Each row shows:
    crop | PCA mask | geometric visibility | intersect (PCA ∧ geometric)

Crops are selected from the same five frames Stage 1 uses (see
``study/templates-v2/geometry/debug_stage1_v2/manifest.json``). Per frame we
pick four pieces spanning piece types and visibility levels so the sheet
covers the full 12 piece classes plus partial-occlusion cases.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
_GEOMETRY_DIR = _THIS_DIR.parent / "geometry"
for _path in (_PROJECT_ROOT, _GEOMETRY_DIR, _THIS_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cv2
import numpy as np

from foreground import (
    DEFAULT_PCA_LAYER_INDEX,
    ForegroundMasks,
    compute_foreground_masks,
    get_default_embedder,
)

from visibility import (
    DEFAULT_INPUT_SIZE,
    board_fen_from_piece_entries,
    compute_frame_visibility,
    detect_piece_board_positions,
    extract_piece_crop,
    occupied_squares_from_fen,
)

DEFAULT_LABELS_PATH = _PROJECT_ROOT / "study" / "eval" / "labels.jsonl"
DEFAULT_MANIFEST_PATH = (
    _PROJECT_ROOT
    / "study"
    / "templates-v2"
    / "geometry"
    / "debug_stage1_v2"
    / "manifest.json"
)
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "study" / "templates-v2" / "isolation" / "preview"

CELL_SIZE = DEFAULT_INPUT_SIZE
CELL_PAD = 12
HEADER_HEIGHT = 42
ROW_LABEL_WIDTH = 110
MASK_ALPHA = 0.40
PIECES_PER_FRAME = 4


def _load_jsonl_row_by_frame_id(labels_path: Path, frame_id: str) -> dict[str, Any]:
    for line in labels_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if row.get("frame_id") == frame_id:
            return row
    raise KeyError(f"frame_id not found in {labels_path}: {frame_id}")


def _select_pieces_for_frame(
    pieces: dict[str, Any],
    used_types: set[str],
    *,
    count: int = PIECES_PER_FRAME,
) -> list[str]:
    """Prefer piece types we have not yet shown, then partial-occlusion cases."""
    items = list(pieces.values())
    items.sort(
        key=lambda piece: (
            0 if piece.symbol not in used_types else 1,
            abs(0.55 - float(piece.visibility_fraction)),
            piece.square,
        )
    )
    selected: list[str] = []
    for piece in items:
        if len(selected) >= count:
            break
        selected.append(piece.square)
    return selected


def _overlay_patch_mask(
    crop_bgr: np.ndarray,
    patch_mask: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise ValueError(f"crop_bgr must be HxWx3, got {crop_bgr.shape}")
    if patch_mask.ndim != 2:
        raise ValueError(f"patch_mask must be 2D, got {patch_mask.shape}")
    upscaled = cv2.resize(
        patch_mask.astype(np.float32),
        (crop_bgr.shape[1], crop_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_3 = np.stack([upscaled] * 3, axis=-1)
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    overlay = crop_bgr.astype(np.float32) * (1.0 - alpha * mask_3) + color * (alpha * mask_3)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _draw_text(canvas: np.ndarray, text: str, anchor: tuple[int, int], *, scale: float = 0.55) -> None:
    x, y = anchor
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _cell_canvas(image: np.ndarray) -> np.ndarray:
    canvas = np.zeros((CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
    canvas[:] = image
    return canvas


def render_contact_sheet(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError("no rows to render")
    column_titles = ("crop", "pca", "geometric", "intersect")
    num_cols = len(column_titles)
    num_rows = len(rows)
    sheet_w = ROW_LABEL_WIDTH + num_cols * CELL_SIZE + (num_cols + 1) * CELL_PAD
    sheet_h = HEADER_HEIGHT + num_rows * CELL_SIZE + (num_rows + 1) * CELL_PAD
    sheet = np.full((sheet_h, sheet_w, 3), 18, dtype=np.uint8)

    for col_index, title in enumerate(column_titles):
        col_x = ROW_LABEL_WIDTH + CELL_PAD + col_index * (CELL_SIZE + CELL_PAD) + 8
        _draw_text(sheet, title.upper(), (col_x, HEADER_HEIGHT - 14), scale=0.7)

    for row_index, row in enumerate(rows):
        y = HEADER_HEIGHT + CELL_PAD + row_index * (CELL_SIZE + CELL_PAD)
        label_lines = [
            f"{row['piece_symbol']} {row['square']}",
            f"vis {row['visibility_fraction']:.2f}",
            row["frame_id"][:24],
        ]
        for line_index, line in enumerate(label_lines):
            _draw_text(sheet, line, (8, y + 22 + line_index * 22), scale=0.45)

        cells = (row["crop"], row["pca_overlay"], row["geom_overlay"], row["intersect_overlay"])
        for col_index, cell in enumerate(cells):
            x = ROW_LABEL_WIDTH + CELL_PAD + col_index * (CELL_SIZE + CELL_PAD)
            sheet[y : y + CELL_SIZE, x : x + CELL_SIZE] = _cell_canvas(cell)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def build_contact_sheet(
    *,
    labels_path: Path = DEFAULT_LABELS_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    layer_index: int = DEFAULT_PCA_LAYER_INDEX,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    embedder = get_default_embedder()

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    used_types: set[str] = set()
    for frame_entry in manifest:
        frame_id = frame_entry["frame_id"]
        label_row = _load_jsonl_row_by_frame_id(labels_path, frame_id)
        image_bgr = cv2.imread(label_row["image_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(label_row["image_path"])
        fen = board_fen_from_piece_entries(label_row["pieces"])
        occupied = occupied_squares_from_fen(fen)
        piece_positions = detect_piece_board_positions(
            image_bgr, label_row["corners"], occupied
        )
        frame_vis = compute_frame_visibility(
            image_shape=image_bgr.shape[:2],
            corners=label_row["corners"],
            fen=fen,
            piece_positions=piece_positions,
            include_full_masks=False,
        )
        squares = _select_pieces_for_frame(frame_vis.pieces, used_types)
        for square in squares:
            piece = frame_vis.pieces[square]
            used_types.add(piece.symbol)
            crop = extract_piece_crop(image_bgr, piece)
            masks = compute_foreground_masks(
                crop, piece, embedder, layer_index=layer_index
            )
            pca_overlay = _overlay_patch_mask(crop, masks.pca, color_bgr=(80, 200, 80))
            geom_overlay = _overlay_patch_mask(
                crop, masks.geometric, color_bgr=(80, 160, 240)
            )
            intersect_overlay = _overlay_patch_mask(
                crop, masks.intersect, color_bgr=(80, 240, 240)
            )
            rows.append(
                {
                    "frame_id": frame_id,
                    "square": square,
                    "piece_symbol": piece.symbol,
                    "visibility_fraction": float(piece.visibility_fraction),
                    "crop": crop,
                    "pca_overlay": pca_overlay,
                    "geom_overlay": geom_overlay,
                    "intersect_overlay": intersect_overlay,
                    "masks": masks,
                }
            )
            summary.append(
                {
                    "frame_id": frame_id,
                    "square": square,
                    "piece_symbol": piece.symbol,
                    "visibility_fraction": float(piece.visibility_fraction),
                    "pca_foreground_patches": int(masks.pca.sum()),
                    "geometric_foreground_patches": int((masks.geometric > 0.3).sum()),
                    "intersect_foreground_patches": int(masks.intersect.sum()),
                }
            )

    render_contact_sheet(rows, output_path=output_dir / "contact_sheet.png")
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--layer-index", type=int, default=DEFAULT_PCA_LAYER_INDEX)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_contact_sheet(
        labels_path=Path(args.labels_path),
        manifest_path=Path(args.manifest_path),
        output_dir=Path(args.output_dir),
        layer_index=int(args.layer_index),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
