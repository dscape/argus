"""Export argus annotations to chesscog's on-disk square-crop layout.

Key deviation from chesscog's own dataset generation: argus stores corners
in board-coordinate order [a8, h8, h1, a1]. Chesscog's `sort_corner_points`
sorts by image coordinates and would scramble this. We feed argus corners
directly into `cv2.findHomography` as the source points, mapping them to
the warped canvas's [TL, TR, BR, BL]. With turn=chess.WHITE throughout,
chesscog's `crop_square` then indexes the canvas correctly because argus's
label order is already canonical white-at-bottom (rank 8 first).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import cv2
import numpy as np
from PIL import Image

from pipeline.physical.chesscog_baseline import (
    ARGUS_CLASS_TO_CHESSCOG_FOLDER,
    ensure_chesscog_on_path,
)
from pipeline.physical.shared.annotation_rows import (
    PhysicalObliqueBoardRow,
    _load_clip_frame_bgr,
    load_annotated_oblique_rows,
)
from pipeline.physical.two_stage.classifier_data import NativeFrameLoader

ensure_chesscog_on_path()

from chesscog.occupancy_classifier.create_dataset import (  # noqa: E402
    BOARD_SIZE as OCC_BOARD_SIZE,
    IMG_SIZE as OCC_IMG_SIZE,
    SQUARE_SIZE as OCC_SQUARE_SIZE,
    crop_square as chesscog_crop_square_occupancy,
)
from chesscog.piece_classifier.create_dataset import (  # noqa: E402
    BOARD_SIZE as PIECE_BOARD_SIZE,
    IMG_SIZE as PIECE_IMG_SIZE,
    MARGIN as PIECE_MARGIN,
    crop_square as chesscog_crop_square_piece,
)


def warp_board_argus_corners(
    rgb: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    *,
    canvas: str,
) -> np.ndarray:
    """Warp an image using argus's board-ordered corners [a8, h8, h1, a1].

    canvas:
      - "occupancy": 500x500 with a 50-px margin (chesscog occupancy canvas)
      - "piece":     800x800 with a 200-px margin (chesscog piece canvas)

    The warped canvas has a8 at TL, h8 at TR, h1 at BR, a1 at BL. chesscog's
    crop_square(turn=chess.WHITE) then indexes this canvas correctly.
    """
    if canvas == "occupancy":
        margin, board_size, img_size = OCC_SQUARE_SIZE, OCC_BOARD_SIZE, OCC_IMG_SIZE
    elif canvas == "piece":
        margin, board_size, img_size = PIECE_MARGIN, PIECE_BOARD_SIZE, PIECE_IMG_SIZE
    else:
        raise ValueError(f"canvas must be 'occupancy' or 'piece', got {canvas!r}")

    src_points = np.asarray(corners, dtype=np.float32)
    if src_points.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {src_points.shape}")

    dst_points = np.array(
        [
            [margin, margin],                       # a8 → TL
            [margin + board_size, margin],          # h8 → TR
            [margin + board_size, margin + board_size],  # h1 → BR
            [margin, margin + board_size],          # a1 → BL
        ],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(rgb, H, (img_size, img_size))


def _resolve_full_frame_and_corners(
    row: PhysicalObliqueBoardRow,
    *,
    native_loader: NativeFrameLoader | None,
    clip_cache: dict,
) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    """Return (BGR frame, corners in frame's coord space).

    Prefers native resolution via NativeFrameLoader when the row has
    native_corners + native_image_bbox + source_video_id (matches argus's
    two-stage eval). Falls back to the clip-frame (224x224) loader.
    """
    if (
        native_loader is not None
        and row.native_corners is not None
        and row.native_image_bbox is not None
        and row.source_video_id is not None
        and row.source_frame_index is not None
    ):
        frame = native_loader.load(
            source_video_id=str(row.source_video_id),
            source_frame_index=int(row.source_frame_index),
        )
        x_off, y_off, _, _ = row.native_image_bbox
        corners = tuple(
            (float(c[0] + x_off), float(c[1] + y_off)) for c in row.native_corners
        )
        return frame, corners

    frame = _load_clip_frame_bgr(row, clip_cache=clip_cache)
    return frame, row.corners


def _square_from_argus_index(i: int) -> chess.Square:
    """argus label index (0..63) → chess.Square. labels[0] = a8, labels[63] = h1."""
    rank = 7 - (i // 8)
    file = i % 8
    return chess.square(file, rank)


def export_split(
    annotation_root: Path,
    output_root: Path,
    split_name: str,
    *,
    limit: int | None = None,
    skip_without_native: bool = True,
) -> dict[str, int]:
    """Produce occupancy + piece crop PNGs under output_root/{occupancy,pieces}/{split_name}/."""
    rows = load_annotated_oblique_rows(annotation_root)
    if skip_without_native:
        rows = [
            r
            for r in rows
            if r.native_corners
            and r.native_image_bbox
            and r.source_video_id
            and r.source_frame_index is not None
        ]
    if limit is not None:
        rows = rows[:limit]
    print(f"[{split_name}] exporting {len(rows)} boards → {output_root}")

    occ_root = output_root / "occupancy" / split_name
    piece_root = output_root / "pieces" / split_name
    for class_name in ("empty", "occupied"):
        (occ_root / class_name).mkdir(parents=True, exist_ok=True)
    for folder in ARGUS_CLASS_TO_CHESSCOG_FOLDER.values():
        (piece_root / folder).mkdir(parents=True, exist_ok=True)

    native_loader = NativeFrameLoader()
    clip_cache: dict = {}

    stats = {
        "boards": 0,
        "occupancy_crops": 0,
        "piece_crops": 0,
        "skipped_rows": 0,
    }
    t0 = time.time()
    try:
        for row_idx, row in enumerate(rows):
            try:
                frame_bgr, corners = _resolve_full_frame_and_corners(
                    row, native_loader=native_loader, clip_cache=clip_cache
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {row.annotation_id}: {exc}")
                stats["skipped_rows"] += 1
                continue

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            warped_occ = warp_board_argus_corners(rgb, corners, canvas="occupancy")
            warped_piece = warp_board_argus_corners(rgb, corners, canvas="piece")

            for i, class_id in enumerate(row.labels):
                square = _square_from_argus_index(i)
                square_name = chess.square_name(square)
                occ_crop = chesscog_crop_square_occupancy(
                    warped_occ, square, chess.WHITE
                )
                occ_folder = "occupied" if class_id != 0 else "empty"
                Image.fromarray(occ_crop, "RGB").save(
                    occ_root / occ_folder / f"{row.annotation_id}_{square_name}.png"
                )
                stats["occupancy_crops"] += 1

                if class_id != 0:
                    piece_folder = ARGUS_CLASS_TO_CHESSCOG_FOLDER[int(class_id)]
                    piece_crop = chesscog_crop_square_piece(
                        warped_piece, square, chess.WHITE
                    )
                    Image.fromarray(piece_crop, "RGB").save(
                        piece_root
                        / piece_folder
                        / f"{row.annotation_id}_{square_name}.png"
                    )
                    stats["piece_crops"] += 1

            stats["boards"] += 1
            if (row_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (row_idx + 1) / max(elapsed, 1e-6)
                print(
                    f"  [{split_name}] {row_idx + 1}/{len(rows)} boards, "
                    f"{rate:.1f} b/s, elapsed {elapsed:.0f}s"
                )
    finally:
        native_loader.close()

    elapsed = time.time() - t0
    print(
        f"[{split_name}] done in {elapsed:.1f}s — "
        f"{stats['boards']} boards, "
        f"{stats['occupancy_crops']} occupancy crops, "
        f"{stats['piece_crops']} piece crops, "
        f"{stats['skipped_rows']} skipped"
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/physical"),
        help="argus physical data root with train/ and val/ subdirs",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/chesscog_baseline"),
        help="where to write chesscog-style crop datasets",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="which argus splits to export",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap rows per split")
    args = ap.parse_args()

    for split in args.splits:
        export_split(
            annotation_root=args.data_root / split,
            output_root=args.output_root,
            split_name=split,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
