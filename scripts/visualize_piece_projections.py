#!/usr/bin/env python3
"""Visualize projected piece bounding boxes across multiple camera angles.

For a sample of physical annotations (ideally spanning different source videos,
hence different camera setups), renders:

1. ``overlay_<id>.jpg``: full native frame with the 64 square base quads (green)
   and projected piece bounding boxes (cyan for empty, orange for occupied).
2. ``piece_<id>_<square>_<class>.jpg``: per-occupied-square piece crop at full
   extraction resolution.

The ``piece_projection`` module adapts automatically to the camera angle
recovered from the 4 corners, so the visualizations show the actual per-frame
projection rather than a one-size-fits-all extension.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.direct_board_reader_data import resolve_source_video_path
from pipeline.physical.oblique_square_context import load_annotated_oblique_rows
from pipeline.physical.piece_projection import (
    camera_pose_from_corners,
    default_camera_matrix,
    extract_projected_piece_crop,
    project_piece_box,
)
from pipeline.shared import SQUARE_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "piece_projection_review"


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir or _DEFAULT_OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_annotated_oblique_rows(args.annotation_root)
    if not rows:
        print(f"no annotations in {args.annotation_root}")
        return

    by_video: dict[str, list] = defaultdict(list)
    for row in rows:
        if row.source_video_id and row.source_frame_index is not None and row.native_corners:
            by_video[row.source_video_id].append(row)
    if not by_video:
        print("no annotations with native-resolution metadata")
        return

    video_caps: dict[Path, cv2.VideoCapture] = {}
    manifest: list[dict] = []
    try:
        for video_id, video_rows in list(by_video.items())[: args.max_videos]:
            first_row = video_rows[0]
            video_path = resolve_source_video_path(video_id)
            cap = video_caps.get(video_path)
            if cap is None:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"failed to open video {video_path}")
                    continue
                video_caps[video_path] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_row.source_frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"failed to read frame {first_row.source_frame_index} from {video_id}")
                continue
            x_off, y_off, _, _ = first_row.native_image_bbox
            full_corners = tuple(
                (float(c[0] + x_off), float(c[1] + y_off)) for c in first_row.native_corners
            )

            overlay_path = output_dir / f"overlay_{first_row.annotation_id}.jpg"
            _render_overlay(
                frame=frame,
                corners=full_corners,
                labels=first_row.labels,
                out_path=overlay_path,
                piece_height=args.piece_height,
            )

            crop_paths: list[str] = []
            for square_index in range(64):
                class_id = int(first_row.labels[square_index])
                if class_id == 0:
                    continue
                row_idx, col_idx = square_index // 8, square_index % 8
                crop = extract_projected_piece_crop(
                    frame,
                    full_corners,
                    row=row_idx,
                    col=col_idx,
                    output_size=args.piece_crop_size,
                    piece_height=args.piece_height,
                )
                square_name = f"{chr(ord('a') + col_idx)}{8 - row_idx}"
                label_name = SQUARE_CLASS_NAMES[class_id]
                crop_path = (
                    output_dir / f"piece_{first_row.annotation_id}_{square_name}_{label_name}.jpg"
                )
                cv2.imwrite(str(crop_path), crop)
                crop_paths.append(str(crop_path.relative_to(_PROJECT_ROOT)))

            manifest.append(
                {
                    "video_id": video_id,
                    "annotation_id": first_row.annotation_id,
                    "overlay": str(overlay_path.relative_to(_PROJECT_ROOT)),
                    "crops": crop_paths,
                }
            )
    finally:
        for cap in video_caps.values():
            cap.release()

    summary_path = output_dir / "summary.txt"
    lines = ["# Piece projection review"]
    for entry in manifest:
        lines.append("")
        lines.append(f"video {entry['video_id']} ({entry['annotation_id']})")
        lines.append(f"  overlay: {entry['overlay']}")
        lines.append(f"  piece crops: {len(entry['crops'])}")
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _render_overlay(
    *,
    frame: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    labels: tuple[int, ...],
    out_path: Path,
    piece_height: float,
) -> None:
    K = default_camera_matrix(frame.shape)
    pose = camera_pose_from_corners(corners, K=K)
    overlay = frame.copy()

    for square_index in range(64):
        row = square_index // 8
        col = square_index % 8
        box = project_piece_box(pose, row=row, col=col, piece_height=piece_height, corners=corners)
        xs, ys = box[:, 0], box[:, 1]
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        class_id = int(labels[square_index])
        color = (180, 180, 180) if class_id == 0 else (40, 180, 240)
        cv2.rectangle(overlay, bbox[:2], bbox[2:], color, 1)
        # Base quad (on the board plane) in green.
        base = box[:4].astype(np.int32)
        cv2.polylines(overlay, [base], isClosed=True, color=(60, 200, 60), thickness=1)
        # Label each square so we can diagnose per-square behavior.
        center = base.mean(axis=0).astype(int)
        square_name = f"{chr(ord('a') + col)}{8 - row}"
        cv2.putText(
            overlay,
            square_name,
            (center[0] - 10, center[1] + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (220, 40, 40),
            1,
            cv2.LINE_AA,
        )

    # Draw the 4 board corners as colored dots for sanity.
    for i, (x, y) in enumerate(corners):
        cv2.circle(overlay, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv2.putText(
            overlay,
            f"c{i}",
            (int(x) + 6, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(str(out_path), overlay)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize piece projection across angles.")
    parser.add_argument(
        "--annotation-root", type=Path, default=_PROJECT_ROOT / "data" / "physical" / "val"
    )
    parser.add_argument("--max-videos", type=int, default=4)
    parser.add_argument("--piece-height", type=float, default=2.0)
    parser.add_argument("--piece-crop-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


if __name__ == "__main__":
    main()
