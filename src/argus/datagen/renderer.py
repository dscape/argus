"""Render loop with ground truth annotation output."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from argus.data.pgn_sampler import generate_game_dataset
from argus.datagen.scene_builder import SceneConfig, compute_table_layout

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    output_dir: str = "data/synthetic"
    resolution: tuple[int, int] = (1920, 1080)
    fps: float = 5.0
    engine: str = "eevee"
    samples: int = 64


@dataclass
class MoveAnnotation:
    move_uci: str
    frame_idx: int
    fen_before: str
    fen_after: str


@dataclass
class BoardAnnotation:
    board_id: int
    bbox_per_frame: list[tuple[float, float, float, float]]
    fen_per_frame: list[str]
    moves: list[MoveAnnotation]


@dataclass
class ClipAnnotation:
    clip_id: str
    num_frames: int
    fps: float
    resolution: tuple[int, int]
    boards: list[BoardAnnotation]


def generate_clip_annotations(
    scene_config: SceneConfig, num_frames: int = 500, fps: float = 5.0, seed: int = 42,
) -> ClipAnnotation:
    import chess
    import random
    rng = random.Random(seed)
    tables = compute_table_layout(scene_config)
    games = generate_game_dataset(num_games=len(tables), min_moves=20, max_moves=80, seed=seed)
    boards: list[BoardAnnotation] = []
    for table, game_moves in zip(tables, games):
        board = chess.Board()
        fens: list[str] = []
        move_anns: list[MoveAnnotation] = []
        avg_interval = num_frames / max(len(game_moves), 1)
        move_idx = 0
        for frame in range(num_frames):
            fens.append(board.fen())
            if move_idx < len(game_moves) and frame >= int(move_idx * avg_interval):
                uci = game_moves[move_idx]
                try:
                    m = chess.Move.from_uci(uci)
                    if m in board.legal_moves:
                        fb = board.fen()
                        board.push(m)
                        move_anns.append(MoveAnnotation(move_uci=uci, frame_idx=frame, fen_before=fb, fen_after=board.fen()))
                except (ValueError, chess.InvalidMoveError):
                    pass
                move_idx += 1
        x, y, _ = table.position
        hw, hd = scene_config.hall_width / 2, scene_config.hall_depth / 2
        cx, cy = (x / hw + 1) / 2, (y / hd + 1) / 2
        bh = 0.03
        bbox = (max(0, cx - bh), max(0, cy - bh), min(1, cx + bh), min(1, cy + bh))
        boards.append(BoardAnnotation(board_id=table.board_id, bbox_per_frame=[bbox] * num_frames, fen_per_frame=fens, moves=move_anns))
    return ClipAnnotation(clip_id=f"clip_{seed:06d}", num_frames=num_frames, fps=fps, resolution=(1920, 1080), boards=boards)


def save_annotations(annotation: ClipAnnotation, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{annotation.clip_id}_annotation.json"
    data = {
        "clip_id": annotation.clip_id, "num_frames": annotation.num_frames,
        "fps": annotation.fps, "resolution": annotation.resolution,
        "boards": [{"board_id": b.board_id, "bbox_per_frame": b.bbox_per_frame, "fen_per_frame": b.fen_per_frame,
                     "moves": [{"move_uci": m.move_uci, "frame_idx": m.frame_idx, "fen_before": m.fen_before, "fen_after": m.fen_after} for m in b.moves]} for b in annotation.boards],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
