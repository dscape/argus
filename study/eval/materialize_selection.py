#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.physical.two_stage.classifier_data import NativeFrameLoader  # noqa: E402
from pipeline.shared import SQUARE_CLASS_NAMES  # noqa: E402

_CATEGORY_NAMES = {
    "a-file-rook",
    "lateral-occlusion",
    "low-camera-angle",
    "dense-middlegame",
    "mid-move",
    "easy-stationary",
}
_VALID_PIECE_NAMES = set(SQUARE_CLASS_NAMES[1:])


def main() -> None:
    args = build_parser().parse_args()
    candidates = load_jsonl_records(args.candidates)
    selections = load_jsonl_records(args.selection)
    candidates_by_id = {str(record["frame_id"]): record for record in candidates}

    output_root = args.output_root.resolve()
    frames_root = output_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    labels_path = output_root / "labels.jsonl"

    emitted: list[dict[str, object]] = []
    seen_frame_ids: set[str] = set()
    loader = NativeFrameLoader(capacity=args.frame_cache_capacity)
    try:
        for selection in selections:
            frame_id = str(selection.get("frame_id", "")).strip()
            if not frame_id:
                raise ValueError(f"selection row is missing frame_id: {selection}")
            if frame_id in seen_frame_ids:
                raise ValueError(f"duplicate frame_id in selection: {frame_id}")
            seen_frame_ids.add(frame_id)

            candidate = candidates_by_id.get(frame_id)
            if candidate is None:
                raise ValueError(f"selection references unknown frame_id: {frame_id}")
            category = str(selection.get("category", "")).strip()
            if category not in _CATEGORY_NAMES:
                raise ValueError(f"invalid category for {frame_id}: {category!r}")

            frame = load_candidate_frame(candidate, native_loader=loader)

            source_video_id = candidate.get("source_video_id")
            source_frame_index = candidate.get("source_frame_index")
            output_image_path = frames_root / f"{frame_id}.jpg"
            if not output_image_path.exists() or args.overwrite:
                ok = cv2.imwrite(str(output_image_path), frame)
                if not ok:
                    raise ValueError(f"failed to write image: {output_image_path}")

            pieces = selection.get("pieces")
            if pieces is None:
                pieces = candidate.get("pieces")
            validated_pieces = validate_piece_payloads(frame_id=frame_id, pieces=pieces)
            notes = selection.get("notes")
            emitted.append(
                {
                    "frame_id": frame_id,
                    "image_path": path_for_storage(output_image_path),
                    "category": category,
                    "corners": candidate["corners"],
                    "pieces": validated_pieces,
                    "source_video_id": source_video_id,
                    "source_frame_index": (
                        None if source_frame_index is None else int(source_frame_index)
                    ),
                    "corner_space": candidate.get("corner_space", "clip_frame"),
                    "notes": None if notes is None else str(notes),
                }
            )
    finally:
        loader.close()

    with labels_path.open("w") as handle:
        for payload in emitted:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    print(f"wrote {len(emitted)} eval labels to {labels_path}")
    print(f"materialized frames under {frames_root}")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"JSONL file not found: {resolved}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(resolved.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid JSON object on line {line_number}: {resolved}")
        records.append(payload)
    return records


def load_candidate_frame(
    candidate: dict[str, Any],
    *,
    native_loader: NativeFrameLoader,
) -> Any:
    source_video_id = candidate.get("source_video_id")
    source_frame_index = candidate.get("source_frame_index")
    if isinstance(source_video_id, str) and source_frame_index is not None:
        return native_loader.load(
            source_video_id=source_video_id,
            source_frame_index=int(source_frame_index),
        )

    clip_path = candidate.get("clip_path")
    frame_index = candidate.get("frame_index")
    if not isinstance(clip_path, str) or frame_index is None:
        raise ValueError(f"candidate is missing native or clip-frame metadata: {candidate}")
    payload = torch.load(
        (_PROJECT_ROOT / clip_path).resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"invalid clip payload: {clip_path}")
    frames = payload.get("frames")
    if not isinstance(frames, torch.Tensor):
        raise ValueError(f"clip is missing frames tensor: {clip_path}")
    frame_tensor = frames[int(frame_index)]
    if frame_tensor.ndim != 3:
        raise ValueError(f"invalid frame tensor shape: {tuple(frame_tensor.shape)}")
    if frame_tensor.shape[0] == 3:
        chw = frame_tensor
    elif frame_tensor.shape[-1] == 3:
        chw = frame_tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"invalid RGB frame tensor shape: {tuple(frame_tensor.shape)}")
    if chw.dtype == torch.uint8:
        rgb = chw.permute(1, 2, 0).cpu().numpy().astype("uint8")
    else:
        rgb = chw.to(torch.float32)
        if float(rgb.max().item()) <= 1.0:
            rgb = rgb * 255.0
        rgb = rgb.clamp(0.0, 255.0).permute(1, 2, 0).cpu().numpy().astype("uint8")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def validate_piece_payloads(
    *,
    frame_id: str,
    pieces: Any,
) -> list[dict[str, object]]:
    if not isinstance(pieces, list):
        raise ValueError(f"{frame_id} has invalid pieces payload")
    validated: list[dict[str, object]] = []
    seen_squares: set[str] = set()
    for piece in pieces:
        if not isinstance(piece, dict):
            raise ValueError(f"{frame_id} has invalid piece payload: {piece!r}")
        piece_type = str(piece.get("type", "")).strip()
        if piece_type not in _VALID_PIECE_NAMES:
            raise ValueError(f"{frame_id} has invalid piece type: {piece_type!r}")
        square = piece.get("square")
        if square is not None:
            square = str(square)
            validate_square_name(square)
            if square in seen_squares:
                raise ValueError(f"{frame_id} repeats occupied square {square}")
            seen_squares.add(square)
        validated.append({"type": piece_type, "square": square})
    return validated


def validate_square_name(square_name: str) -> None:
    if len(square_name) != 2 or square_name[0] < "a" or square_name[0] > "h":
        raise ValueError(f"invalid square name: {square_name!r}")
    rank = int(square_name[1])
    if rank < 1 or rank > 8:
        raise ValueError(f"invalid square name: {square_name!r}")


def path_for_storage(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize study/eval labels + frames.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=_PROJECT_ROOT / "study" / "eval" / "candidates.jsonl",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=_PROJECT_ROOT / "study" / "eval" / "selection.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_PROJECT_ROOT / "study" / "eval",
    )
    parser.add_argument("--frame-cache-capacity", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    main()
