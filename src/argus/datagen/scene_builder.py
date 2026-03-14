"""Blender scene composition for tournament hall rendering."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class TableConfig:
    position: tuple[float, float, float]
    rotation: float
    board_id: int


@dataclass
class SceneConfig:
    num_tables: int = 20
    hall_width: float = 20.0
    hall_depth: float = 30.0
    hall_height: float = 4.0
    table_spacing: float = 2.5
    table_height: float = 0.75
    board_size: float = 0.45
    seed: int = 42


def compute_table_layout(config: SceneConfig) -> list[TableConfig]:
    rng = random.Random(config.seed)
    tables: list[TableConfig] = []
    cols = max(1, int(config.hall_width / config.table_spacing))
    rows = max(1, int(config.hall_depth / config.table_spacing))
    board_id = 0
    for row in range(rows):
        for col in range(cols):
            if board_id >= config.num_tables:
                break
            x = (col - cols / 2 + 0.5) * config.table_spacing + rng.uniform(-0.1, 0.1)
            y = (row - rows / 2 + 0.5) * config.table_spacing + rng.uniform(-0.1, 0.1)
            z = config.table_height
            rotation = rng.uniform(-0.05, 0.05)
            tables.append(TableConfig(position=(x, y, z), rotation=rotation, board_id=board_id))
            board_id += 1
    return tables


def build_scene_blender(config: SceneConfig) -> None:
    try:
        import bpy
    except ImportError:
        raise RuntimeError("Must run inside Blender's Python environment.")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_plane_add(size=max(config.hall_width, config.hall_depth) * 1.5, location=(0, 0, 0))
    bpy.context.active_object.name = "Floor"
    tables = compute_table_layout(config)
    for table in tables:
        x, y, z = table.position
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z))
        obj = bpy.context.active_object
        obj.name = f"Table_{table.board_id}"
        obj.scale = (0.8, 0.5, 0.02)
        obj.rotation_euler[2] = table.rotation
        bpy.ops.mesh.primitive_plane_add(size=config.board_size, location=(x, y, z + 0.025))
        bpy.context.active_object.name = f"Board_{table.board_id}"
