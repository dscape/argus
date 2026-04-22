#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.physical.piece_projection import (  # noqa: E402
    camera_pose_from_corners,
    default_camera_matrix,
    project_piece_bboxes,
)

_CATEGORY_ORDER = (
    "mid-move",
    "a-file-rook",
    "lateral-occlusion",
    "dense-middlegame",
    "low-camera-angle",
    "easy-stationary",
)
_DEFAULT_COUNTS = {
    "a-file-rook": 50,
    "lateral-occlusion": 50,
    "low-camera-angle": 50,
    "dense-middlegame": 50,
    "mid-move": 50,
    "easy-stationary": 100,
}


@dataclass(frozen=True)
class CandidateFeatures:
    frame_id: str
    clip_path: str
    frame_index: int
    candidate_type: str
    occupied_count: int
    elevation_deg: float
    contamination: float
    a_file_rook_score: float
    lateral_score: float
    dense_score: float
    easy_score: float

    @property
    def key(self) -> tuple[str, int]:
        return (self.clip_path, self.frame_index)


def main() -> None:
    args = build_parser().parse_args()
    candidates = load_candidates(args.candidates)
    features = {candidate["frame_id"]: compute_features(candidate) for candidate in candidates}

    requested_counts = {
        "a-file-rook": args.a_file_rook_count,
        "lateral-occlusion": args.lateral_occlusion_count,
        "low-camera-angle": args.low_camera_angle_count,
        "dense-middlegame": args.dense_middlegame_count,
        "mid-move": args.mid_move_count,
        "easy-stationary": args.easy_stationary_count,
    }

    selections, summary = auto_select_candidates(
        candidates,
        features=features,
        requested_counts=requested_counts,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for payload in selections:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    summary_path = args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"wrote {len(selections)} selections to {output_path}")
    print(json.dumps(summary["selected_counts"], indent=2, sort_keys=True))


def auto_select_candidates(
    candidates: list[dict[str, Any]],
    *,
    features: dict[str, CandidateFeatures],
    requested_counts: dict[str, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    board_candidates = [
        candidate for candidate in candidates if candidate.get("candidate_type") == "board"
    ]
    transient_candidates = [
        candidate for candidate in candidates if candidate.get("candidate_type") == "transient"
    ]
    candidates_by_category = {
        "mid-move": transient_candidates,
        "a-file-rook": board_candidates,
        "lateral-occlusion": board_candidates,
        "dense-middlegame": board_candidates,
        "low-camera-angle": board_candidates,
        "easy-stationary": board_candidates,
    }

    used_keys: set[tuple[str, int]] = set()
    selections: list[dict[str, object]] = []
    selected_counts: dict[str, int] = {}
    shortages: dict[str, int] = {}

    for category in _CATEGORY_ORDER:
        category_candidates = rank_candidates(
            category,
            candidates_by_category[category],
            features=features,
        )
        picked = pick_candidates(
            category,
            category_candidates,
            features=features,
            used_keys=used_keys,
            target_count=requested_counts[category],
        )
        for candidate in picked:
            used_keys.add(features[candidate["frame_id"]].key)
            selections.append(
                {
                    "frame_id": candidate["frame_id"],
                    "category": category,
                    "notes": auto_note(category, candidate, features[candidate["frame_id"]]),
                }
            )
        selected_counts[category] = len(picked)
        shortages[category] = max(requested_counts[category] - len(picked), 0)

    summary = {
        "requested_counts": requested_counts,
        "selected_counts": selected_counts,
        "shortages": shortages,
        "selected_total": len(selections),
        "selected_by_clip": dict(Counter(selection_clip_path(selections, candidates))),
    }
    return selections, summary


def selection_clip_path(
    selections: list[dict[str, object]],
    candidates: list[dict[str, Any]],
) -> list[str]:
    candidates_by_frame_id = {str(candidate["frame_id"]): candidate for candidate in candidates}
    clip_paths: list[str] = []
    for selection in selections:
        candidate = candidates_by_frame_id[str(selection["frame_id"])]
        clip_paths.append(str(candidate["clip_path"]))
    return clip_paths


def pick_candidates(
    category: str,
    ranked_candidates: list[dict[str, Any]],
    *,
    features: dict[str, CandidateFeatures],
    used_keys: set[tuple[str, int]],
    target_count: int,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []

    configs = category_pick_configs(category)
    for min_gap, max_per_clip in configs:
        picked = pick_with_constraints(
            ranked_candidates,
            features=features,
            used_keys=used_keys,
            target_count=target_count,
            min_frame_gap=min_gap,
            max_per_clip=max_per_clip,
        )
        if len(picked) >= target_count:
            return picked
    return picked


def pick_with_constraints(
    ranked_candidates: list[dict[str, Any]],
    *,
    features: dict[str, CandidateFeatures],
    used_keys: set[tuple[str, int]],
    target_count: int,
    min_frame_gap: int,
    max_per_clip: int | None,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    picked_frames_by_clip: dict[str, list[int]] = defaultdict(list)
    clip_counts: Counter[str] = Counter()
    for candidate in ranked_candidates:
        candidate_features = features[candidate["frame_id"]]
        key = candidate_features.key
        if key in used_keys:
            continue
        clip_path = candidate_features.clip_path
        frame_index = candidate_features.frame_index
        if max_per_clip is not None and clip_counts[clip_path] >= max_per_clip:
            continue
        if any(
            abs(frame_index - chosen) < min_frame_gap
            for chosen in picked_frames_by_clip[clip_path]
        ):
            continue
        picked.append(candidate)
        picked_frames_by_clip[clip_path].append(frame_index)
        clip_counts[clip_path] += 1
        if len(picked) >= target_count:
            break
    return picked


def category_pick_configs(category: str) -> list[tuple[int, int | None]]:
    if category == "easy-stationary":
        return [(1, 100), (1, None)]
    if category == "mid-move":
        return [(1, None)]
    return [(4, 20), (2, 25), (1, None)]


def rank_candidates(
    category: str,
    candidates: list[dict[str, Any]],
    *,
    features: dict[str, CandidateFeatures],
) -> list[dict[str, Any]]:
    filtered = [
        candidate
        for candidate in candidates
        if passes_category_filter(category, features[candidate["frame_id"]])
    ]
    return sorted(
        filtered,
        key=lambda candidate: category_sort_key(category, features[candidate["frame_id"]]),
    )


def passes_category_filter(category: str, feature: CandidateFeatures) -> bool:
    if category == "mid-move":
        return feature.candidate_type == "transient"
    if category == "a-file-rook":
        return feature.a_file_rook_score > 0.0
    if category == "lateral-occlusion":
        return feature.lateral_score > 0.0
    if category == "dense-middlegame":
        return feature.occupied_count >= 24
    if category == "low-camera-angle":
        return feature.candidate_type == "board"
    if category == "easy-stationary":
        return feature.candidate_type == "board" and feature.easy_score > 0.0
    return False


def category_sort_key(category: str, feature: CandidateFeatures) -> tuple[float, ...]:
    if category == "mid-move":
        return (feature.frame_index,)
    if category == "a-file-rook":
        return (-feature.a_file_rook_score, -feature.contamination, -feature.occupied_count)
    if category == "lateral-occlusion":
        return (-feature.lateral_score, -feature.contamination, -feature.occupied_count)
    if category == "dense-middlegame":
        return (-feature.dense_score, -feature.contamination, feature.elevation_deg)
    if category == "low-camera-angle":
        return (feature.elevation_deg, -feature.contamination, -feature.occupied_count)
    if category == "easy-stationary":
        return (-feature.easy_score, -feature.elevation_deg, feature.contamination)
    raise ValueError(f"unsupported category: {category}")


def auto_note(category: str, candidate: dict[str, Any], feature: CandidateFeatures) -> str:
    if category == "mid-move":
        return (
            f"auto-selected transient span for {candidate.get('transient_move_uci')} "
            f"at frame {candidate['frame_index']}"
        )
    if category == "a-file-rook":
        return f"auto-selected edge-rook overlap score={feature.a_file_rook_score:.3f}"
    if category == "lateral-occlusion":
        return f"auto-selected lateral overlap score={feature.lateral_score:.3f}"
    if category == "dense-middlegame":
        return (
            f"auto-selected dense board pieces={feature.occupied_count} "
            f"dense_score={feature.dense_score:.3f}"
        )
    if category == "low-camera-angle":
        return f"auto-selected low camera elevation={feature.elevation_deg:.2f}"
    if category == "easy-stationary":
        return f"auto-selected easy/stationary score={feature.easy_score:.3f}"
    return "auto-selected"


def load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Candidates file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def compute_features(candidate: dict[str, Any]) -> CandidateFeatures:
    pieces = candidate.get("pieces")
    if not isinstance(pieces, list):
        raise ValueError(f"candidate has invalid pieces payload: {candidate}")
    occupied = [piece for piece in pieces if piece.get("square") is not None]
    occupied_count = len(occupied)
    if candidate.get("candidate_type") == "transient":
        return CandidateFeatures(
            frame_id=str(candidate["frame_id"]),
            clip_path=str(candidate["clip_path"]),
            frame_index=int(candidate["frame_index"]),
            candidate_type=str(candidate["candidate_type"]),
            occupied_count=occupied_count,
            elevation_deg=0.0,
            contamination=0.0,
            a_file_rook_score=0.0,
            lateral_score=0.0,
            dense_score=float(occupied_count),
            easy_score=0.0,
        )

    corners = tuple((float(x), float(y)) for x, y in candidate["corners"])
    pose = camera_pose_from_corners(corners, K=default_camera_matrix((1080, 1920, 3)))
    camera_position = (-pose.R.T @ pose.t).reshape(-1)
    elevation_deg = abs(
        math.degrees(
            math.atan2(
                float(camera_position[2]),
                math.hypot(float(camera_position[0]), float(camera_position[1])),
            )
        )
    )
    projected_bboxes = project_piece_bboxes(corners, frame_shape=(1080, 1920, 3))

    occupied_square_indices = [square_name_to_index(str(piece["square"])) for piece in occupied]
    piece_by_index = {
        square_name_to_index(str(piece["square"])): str(piece["type"]) for piece in occupied
    }

    contamination = 0.0
    a_file_rook_score = 0.0
    lateral_score = 0.0
    for square_index, piece_type in piece_by_index.items():
        piece_bbox = projected_bboxes[square_index]
        contamination = max(
            contamination,
            max_empty_square_overlap(piece_bbox, projected_bboxes, occupied_square_indices),
        )
        a_file_rook_score = max(
            a_file_rook_score,
            edge_rook_overlap_score(
                piece_type,
                square_index,
                projected_bboxes=projected_bboxes,
            ),
        )
        lateral_score = max(
            lateral_score,
            knight_bishop_lateral_score(
                piece_type,
                square_index,
                occupied_square_indices=occupied_square_indices,
                projected_bboxes=projected_bboxes,
            ),
        )

    dense_score = float(occupied_count) + (contamination * 10.0)
    easy_score = (
        elevation_deg * 1.5
        - (contamination * 10.0)
        - (a_file_rook_score * 4.0)
        - (lateral_score * 4.0)
        - max(occupied_count - 24, 0)
    )
    return CandidateFeatures(
        frame_id=str(candidate["frame_id"]),
        clip_path=str(candidate["clip_path"]),
        frame_index=int(candidate["frame_index"]),
        candidate_type=str(candidate["candidate_type"]),
        occupied_count=occupied_count,
        elevation_deg=elevation_deg,
        contamination=contamination,
        a_file_rook_score=a_file_rook_score,
        lateral_score=lateral_score,
        dense_score=dense_score,
        easy_score=easy_score,
    )


def max_empty_square_overlap(
    piece_bbox: Any,
    projected_bboxes: Any,
    occupied_square_indices: list[int],
) -> float:
    overlap = 0.0
    occupied_set = set(occupied_square_indices)
    for square_index in range(64):
        if square_index in occupied_set:
            continue
        overlap = max(overlap, bbox_overlap_ratio(projected_bboxes[square_index], piece_bbox))
    return overlap


def edge_rook_overlap_score(
    piece_type: str,
    square_index: int,
    *,
    projected_bboxes: Any,
) -> float:
    if piece_type.lower() != "r":
        return 0.0
    square_name = index_to_square_name(square_index)
    if square_name not in {"a1", "a8", "h1", "h8"}:
        return 0.0
    file_name = square_name[0]
    rank = int(square_name[1])
    ahead_squares = [f"{file_name}{next_rank}" for next_rank in range(rank + 1, min(9, rank + 4))]
    score = 0.0
    for ahead_square in ahead_squares:
        score = max(
            score,
            bbox_overlap_ratio(
                projected_bboxes[square_name_to_index(ahead_square)],
                projected_bboxes[square_index],
            ),
        )
    return score


def knight_bishop_lateral_score(
    piece_type: str,
    square_index: int,
    *,
    occupied_square_indices: list[int],
    projected_bboxes: Any,
) -> float:
    if piece_type.lower() not in {"n", "b"}:
        return 0.0
    row_index, file_index = divmod(square_index, 8)
    score = 0.0
    for other_square_index in occupied_square_indices:
        if other_square_index == square_index:
            continue
        other_row_index, other_file_index = divmod(other_square_index, 8)
        if abs(other_file_index - file_index) != 1:
            continue
        if abs(other_row_index - row_index) > 2:
            continue
        score = max(
            score,
            bbox_overlap_ratio(
                projected_bboxes[other_square_index],
                projected_bboxes[square_index],
            ),
        )
    return score


def bbox_overlap_ratio(target_bbox: Any, foreign_bbox: Any) -> float:
    tx1, ty1, tx2, ty2 = [float(value) for value in target_bbox.tolist()]
    fx1, fy1, fx2, fy2 = [float(value) for value in foreign_bbox.tolist()]
    inter_x1 = max(tx1, fx1)
    inter_y1 = max(ty1, fy1)
    inter_x2 = min(tx2, fx2)
    inter_y2 = min(ty2, fy2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    if target_area <= 1e-6:
        return 0.0
    return (inter_w * inter_h) / target_area


def square_name_to_index(square_name: str) -> int:
    file_index = ord(square_name[0]) - ord("a")
    rank = int(square_name[1])
    return (8 - rank) * 8 + file_index


def index_to_square_name(square_index: int) -> str:
    row_index = square_index // 8
    file_index = square_index % 8
    rank = 8 - row_index
    return f"{chr(ord('a') + file_index)}{rank}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-select the study/eval frame breakdown.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=_PROJECT_ROOT / "study" / "eval" / "candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "study" / "eval" / "selection.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=_PROJECT_ROOT / "study" / "eval" / "selection_summary.json",
    )
    parser.add_argument("--a-file-rook-count", type=int, default=_DEFAULT_COUNTS["a-file-rook"])
    parser.add_argument(
        "--lateral-occlusion-count",
        type=int,
        default=_DEFAULT_COUNTS["lateral-occlusion"],
    )
    parser.add_argument(
        "--low-camera-angle-count",
        type=int,
        default=_DEFAULT_COUNTS["low-camera-angle"],
    )
    parser.add_argument(
        "--dense-middlegame-count",
        type=int,
        default=_DEFAULT_COUNTS["dense-middlegame"],
    )
    parser.add_argument("--mid-move-count", type=int, default=_DEFAULT_COUNTS["mid-move"])
    parser.add_argument(
        "--easy-stationary-count",
        type=int,
        default=_DEFAULT_COUNTS["easy-stationary"],
    )
    return parser


if __name__ == "__main__":
    main()
