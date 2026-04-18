#!/usr/bin/env python3
"""Export replay-derived real physical boards for tomorrow's iteration."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.physical.shared.eval_dataset import DEFAULT_BOARD_SIZE, rectify_board_image
from pipeline.physical.shared.real_board_data import PhysicalRealBoardDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "physical_real_board_dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replay-derived real physical boards")
    parser.add_argument("--clips-dir", type=Path, default=Path("data/argus/train_real"))
    parser.add_argument("--eval-root", type=Path, default=Path("data/physical/eval"))
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--exclude-move-neighborhood", type=int, default=-1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-size", type=int, default=DEFAULT_BOARD_SIZE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    boards_dir = output_dir / "boards"
    boards_dir.mkdir(parents=True, exist_ok=True)

    dataset = PhysicalRealBoardDataset(
        clips_dir=args.clips_dir,
        eval_root=args.eval_root,
        frame_stride=args.frame_stride,
        max_frames=(None if args.max_frames <= 0 else args.max_frames),
        exclude_move_neighborhood=args.exclude_move_neighborhood,
    )

    manifest_rows: list[dict[str, Any]] = []
    clip_cache: dict[Path, dict[str, Any]] = {}
    for index, row in enumerate(dataset.rows):
        clip = load_clip(clip_cache, _PROJECT_ROOT / row.clip_path)
        frames = clip.get("frames")
        if not isinstance(frames, torch.Tensor):
            continue
        frame = frames[row.frame_index]
        image_rgb = frame_tensor_to_rgb(frame)
        rectified_rgb = rectify_board_image(
            image_rgb,
            list(row.corners),
            output_size=args.output_size,
        )
        board_path = boards_dir / f"{index:06d}.jpg"
        if not cv2.imwrite(str(board_path), cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2BGR)):
            raise ValueError(f"Failed to write {board_path}")

        manifest_row = asdict(row)
        manifest_row["board_path"] = str(board_path.resolve().relative_to(_PROJECT_ROOT.resolve()))
        manifest_rows.append(manifest_row)

    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    summary = {
        "row_count": len(manifest_rows),
        "frame_stride": args.frame_stride,
        "exclude_move_neighborhood": args.exclude_move_neighborhood,
        "output_size": args.output_size,
        "source_video_counts": dict(Counter(row["source_video_id"] for row in manifest_rows)),
        "source_channel_counts": dict(
            Counter(row["source_channel_handle"] for row in manifest_rows)
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return _DEFAULT_OUTPUT_ROOT.resolve()


def load_clip(cache: dict[Path, dict[str, Any]], clip_path: Path) -> dict[str, Any]:
    cached = cache.get(clip_path)
    if cached is not None:
        return cached
    clip = torch.load(clip_path, map_location="cpu", weights_only=False)
    if not isinstance(clip, dict):
        raise ValueError(f"Invalid clip file: {clip_path}")
    cache[clip_path] = clip
    return clip


def frame_tensor_to_rgb(frame: torch.Tensor) -> Any:
    if frame.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB frame, got {tuple(frame.shape)}")
    if frame.dtype == torch.uint8:
        return frame.permute(1, 2, 0).cpu().numpy()
    rgb = frame.to(torch.float32)
    if float(rgb.max().item()) <= 1.0:
        rgb = rgb * 255.0
    return rgb.clamp(0.0, 255.0).permute(1, 2, 0).cpu().numpy().astype("uint8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


if __name__ == "__main__":
    main()
