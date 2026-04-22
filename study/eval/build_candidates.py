#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chess
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.physical.shared.annotation_rows import load_annotated_oblique_rows  # noqa: E402
from pipeline.physical.shared.real_board_data import infer_channel_corner_templates  # noqa: E402
from pipeline.shared import SQUARE_CLASS_NAMES  # noqa: E402


def main() -> None:
    args = build_parser().parse_args()
    board_candidates = build_board_candidates(args.annotation_root)
    transient_candidates = build_transient_candidates(args.annotation_root)
    candidates = board_candidates + transient_candidates
    candidates.sort(key=lambda payload: (str(payload["clip_path"]), int(payload["frame_index"])))
    if args.limit > 0:
        candidates = candidates[: args.limit]

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for payload in candidates:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    print(
        f"wrote {len(candidates)} candidates to {output_path} "
        f"({len(board_candidates)} board + {len(transient_candidates)} transient)"
    )


def build_board_candidates(annotation_root: Path) -> list[dict[str, object]]:
    rows = load_annotated_oblique_rows(annotation_root)
    return [build_board_candidate_payload(row) for row in rows if has_native_metadata(row)]


def build_transient_candidates(annotation_root: Path) -> list[dict[str, object]]:
    board_rows_by_clip_frame = load_raw_board_rows_by_clip_frame(annotation_root)
    transient_annotations = load_jsonl(annotation_root / "transient_annotations.jsonl")
    train_root = annotation_root.parent / "train"
    channel_templates = infer_channel_corner_templates(eval_root=train_root)
    channel_templates.update(infer_channel_corner_templates(eval_root=annotation_root))
    clip_cache: dict[Path, dict[str, Any]] = {}
    candidates: list[dict[str, object]] = []
    seen_frame_ids: set[str] = set()

    for annotation in transient_annotations:
        clip_path = annotation.get("clip_path")
        move_annotations = annotation.get("move_annotations")
        if not isinstance(clip_path, str) or not isinstance(move_annotations, list):
            continue
        clip_payload = load_clip_payload(Path(clip_path), clip_cache=clip_cache)
        for move_annotation in move_annotations:
            start_frame_index = move_annotation.get("start_frame_index")
            end_frame_index = move_annotation.get("end_frame_index")
            if start_frame_index is None or end_frame_index is None:
                continue
            source_piece_payloads = transient_piece_payloads(move_annotation)
            for frame_index in range(int(start_frame_index), int(end_frame_index) + 1):
                raw_row = board_rows_by_clip_frame.get((clip_path, frame_index))
                if raw_row is None:
                    raw_row = nearest_raw_row(
                        board_rows_by_clip_frame,
                        clip_path=clip_path,
                        frame_index=frame_index,
                    )
                frame_context = frame_context_for_transient_candidate(
                    clip_path=clip_path,
                    frame_index=frame_index,
                    raw_row=raw_row,
                    clip_payload=clip_payload,
                    channel_templates=channel_templates,
                )
                if frame_context is None:
                    continue
                frame_id = f"{Path(clip_path).stem}_frame{frame_index:04d}_transient"
                if frame_id in seen_frame_ids:
                    continue
                seen_frame_ids.add(frame_id)
                candidates.append(
                    build_transient_candidate_payload(
                        frame_id=frame_id,
                        clip_path=clip_path,
                        frame_index=frame_index,
                        frame_context=frame_context,
                        pieces=source_piece_payloads,
                        move_annotation=move_annotation,
                    )
                )
    return candidates


def has_native_metadata(row: Any) -> bool:
    return (
        row.source_video_id is not None
        and row.source_frame_index is not None
        and row.native_corners is not None
        and row.native_image_bbox is not None
    )


def build_board_candidate_payload(row: Any) -> dict[str, object]:
    if row.native_corners is None or row.native_image_bbox is None:
        raise ValueError(f"row is missing native corner metadata: {row.annotation_id}")
    x_off, y_off, _width, _height = row.native_image_bbox
    full_frame_corners = [[float(x + x_off), float(y + y_off)] for x, y in row.native_corners]
    frame_id = str(row.annotation_id)
    return {
        "frame_id": frame_id,
        "annotation_id": str(row.annotation_id),
        "candidate_type": "board",
        "clip_path": str(row.clip_path),
        "frame_index": int(row.frame_index),
        "source_video_id": str(row.source_video_id),
        "source_frame_index": int(row.source_frame_index),
        "corners": full_frame_corners,
        "corner_space": "native_frame",
        "pieces": labels_to_piece_payloads(row.labels),
        "materialized_image_path": f"study/eval/frames/{frame_id}.jpg",
    }


def build_transient_candidate_payload(
    *,
    frame_id: str,
    clip_path: str,
    frame_index: int,
    frame_context: dict[str, Any],
    pieces: list[dict[str, object]],
    move_annotation: dict[str, Any],
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "annotation_id": frame_id,
        "candidate_type": "transient",
        "clip_path": clip_path,
        "frame_index": int(frame_index),
        "source_video_id": frame_context.get("source_video_id"),
        "source_frame_index": frame_context.get("source_frame_index"),
        "corners": frame_context["corners"],
        "corner_space": frame_context["corner_space"],
        "pieces": pieces,
        "transient_move_uci": str(move_annotation["uci"]),
        "transient_start_frame_index": int(move_annotation["start_frame_index"]),
        "transient_move_frame_index": int(move_annotation["move_frame_index"]),
        "transient_end_frame_index": int(move_annotation["end_frame_index"]),
        "materialized_image_path": f"study/eval/frames/{frame_id}.jpg",
    }


def load_raw_board_rows_by_clip_frame(
    annotation_root: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for payload in load_jsonl(annotation_root / "board_annotations.jsonl"):
        clip_path = payload.get("clip_path")
        frame_index = payload.get("frame_index")
        if not isinstance(clip_path, str) or frame_index is None:
            continue
        if not is_valid_raw_corner_row(payload):
            continue
        rows[(clip_path, int(frame_index))] = payload
    return rows


def nearest_raw_row(
    rows_by_clip_frame: dict[tuple[str, int], dict[str, Any]],
    *,
    clip_path: str,
    frame_index: int,
) -> dict[str, Any] | None:
    candidates = [
        (abs(candidate_frame_index - frame_index), payload)
        for (candidate_clip_path, candidate_frame_index), payload in rows_by_clip_frame.items()
        if candidate_clip_path == clip_path
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def is_valid_raw_corner_row(payload: dict[str, Any]) -> bool:
    raw_native_corners = payload.get("native_corners")
    raw_native_image_bbox = payload.get("native_image_bbox")
    raw_corners = payload.get("corners")
    has_native = (
        isinstance(raw_native_corners, list)
        and len(raw_native_corners) == 4
        and isinstance(raw_native_image_bbox, list)
        and len(raw_native_image_bbox) == 4
    )
    has_clip_frame = isinstance(raw_corners, list) and len(raw_corners) == 4
    return has_native or has_clip_frame


def full_frame_corners_from_raw_row(raw_row: dict[str, Any]) -> list[list[float]]:
    raw_native_corners = raw_row["native_corners"]
    raw_native_image_bbox = raw_row["native_image_bbox"]
    x_off, y_off, _width, _height = [int(value) for value in raw_native_image_bbox]
    return [
        [float(point[0] + x_off), float(point[1] + y_off)]
        for point in raw_native_corners
    ]


def clip_frame_corners_from_raw_row(raw_row: dict[str, Any]) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in raw_row["corners"]]


def frame_context_for_transient_candidate(
    *,
    clip_path: str,
    frame_index: int,
    raw_row: dict[str, Any] | None,
    clip_payload: dict[str, Any],
    channel_templates: dict[str, tuple[tuple[float, float], ...]],
) -> dict[str, Any] | None:
    if raw_row is not None:
        has_native = (
            raw_row.get("native_corners") is not None
            and raw_row.get("native_image_bbox") is not None
        )
        if has_native:
            return {
                "corners": full_frame_corners_from_raw_row(raw_row),
                "corner_space": "native_frame",
                "source_video_id": raw_row.get("source_video_id"),
                "source_frame_index": raw_row.get("source_frame_index"),
            }
        if isinstance(raw_row.get("corners"), list) and len(raw_row["corners"]) == 4:
            return {
                "corners": clip_frame_corners_from_raw_row(raw_row),
                "corner_space": "clip_frame",
                "source_video_id": raw_row.get("source_video_id"),
                "source_frame_index": raw_row.get("source_frame_index"),
            }

    source_channel_handle = clip_payload.get("source_channel_handle")
    if not isinstance(source_channel_handle, str):
        return None
    template_corners = channel_templates.get(source_channel_handle)
    if template_corners is None:
        return None
    frame_indices = clip_payload.get("frame_indices")
    source_frame_index = None
    if isinstance(frame_indices, torch.Tensor) and 0 <= frame_index < int(frame_indices.shape[0]):
        source_frame_index = int(frame_indices[frame_index].item())
    return {
        "corners": [[float(x), float(y)] for x, y in template_corners],
        "corner_space": "clip_frame",
        "source_video_id": (
            clip_payload.get("source_video_id")
            if isinstance(clip_payload.get("source_video_id"), str)
            else None
        ),
        "source_frame_index": source_frame_index,
    }


def load_clip_payload(clip_path: Path, *, clip_cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    resolved = (_PROJECT_ROOT / clip_path).resolve()
    cached = clip_cache.get(resolved)
    if cached is not None:
        return cached
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid clip payload: {resolved}")
    clip_cache[resolved] = payload
    return payload


def transient_piece_payloads(move_annotation: dict[str, Any]) -> list[dict[str, object]]:
    fen_before = move_annotation.get("fen_before")
    uci = move_annotation.get("uci")
    if not isinstance(fen_before, str) or not isinstance(uci, str):
        raise ValueError(f"invalid transient move annotation: {move_annotation}")
    board = chess.Board(fen_before)
    move = chess.Move.from_uci(uci)
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError(f"fen_before has no moving piece for {uci}: {fen_before}")

    pieces: list[dict[str, object]] = []
    for square, piece in board.piece_map().items():
        if square == move.from_square:
            continue
        pieces.append(
            {
                "type": piece.symbol(),
                "square": chess.square_name(square),
            }
        )
    pieces.append(
        {
            "type": moving_piece.symbol(),
            "square": None,
        }
    )
    pieces.sort(key=lambda payload: (str(payload["type"]), str(payload["square"])))
    return pieces


def labels_to_piece_payloads(labels: tuple[int, ...]) -> list[dict[str, object]]:
    if len(labels) != 64:
        raise ValueError(f"expected 64 labels, got {len(labels)}")
    pieces: list[dict[str, object]] = []
    for square_index, label in enumerate(labels):
        if label <= 0:
            continue
        pieces.append(
            {
                "type": SQUARE_CLASS_NAMES[label],
                "square": index_to_square_name(square_index),
            }
        )
    return pieces


def index_to_square_name(square_index: int) -> str:
    if square_index < 0 or square_index >= 64:
        raise ValueError(f"square_index out of range: {square_index}")
    row_index = square_index // 8
    file_index = square_index % 8
    rank = 8 - row_index
    return f"{chr(ord('a') + file_index)}{rank}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build study/eval candidate rows from held-out physical annotations."
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "physical" / "val",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "study" / "eval" / "candidates.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser


if __name__ == "__main__":
    main()
