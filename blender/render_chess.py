"""Blender Python script for realistic chess board rendering.

Run inside Blender:
    blender --background --python render_chess.py -- \
        --manifest /tmp/clip_manifest.json \
        --output-dir /tmp/clip_frames/ \
        --resolution 448

Reads a JSON manifest with board theme, piece material, lighting, and
a list of frames (FEN + camera angles). Renders each frame to PNG.
"""

import json
import math
import os
import sys
from pathlib import Path

import bpy
import mathutils

# ---------------------------------------------------------------------------
# Blender args parsing (everything after --)
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resolution", type=int, default=448)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Scene cleanup
# ---------------------------------------------------------------------------

def clear_scene():
    """Remove all objects, materials, and meshes from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


# ---------------------------------------------------------------------------
# Material creation
# ---------------------------------------------------------------------------

def hex_to_rgba(hex_color: str) -> tuple:
    """Convert hex color to RGBA for Blender's Principled BSDF Base Color."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def rgb_to_rgba(r: int, g: int, b: int) -> tuple:
    """Convert RGB 0-255 to RGBA for Blender."""
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


def create_piece_material(name: str, color: list, mat_type: str) -> bpy.types.Material:
    """Create a Principled BSDF material for a chess piece."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = rgb_to_rgba(*color)

    if mat_type == "plastic":
        bsdf.inputs["Roughness"].default_value = 0.35
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["IOR"].default_value = 1.46
    elif mat_type == "wood":
        bsdf.inputs["Roughness"].default_value = 0.50
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["IOR"].default_value = 1.5
        # Add subtle subsurface for wood warmth
        bsdf.inputs["Subsurface Weight"].default_value = 0.05
    elif mat_type == "metal":
        bsdf.inputs["Roughness"].default_value = 0.15
        bsdf.inputs["Metallic"].default_value = 0.9
        bsdf.inputs["IOR"].default_value = 2.5

    return mat


def create_board_material(name: str, hex_color: str, texture_type: str) -> bpy.types.Material:
    """Create a material for a board square."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = hex_to_rgba(hex_color)

    if texture_type == "vinyl":
        bsdf.inputs["Roughness"].default_value = 0.65
        bsdf.inputs["Metallic"].default_value = 0.0
    elif texture_type == "wood":
        bsdf.inputs["Roughness"].default_value = 0.45
        bsdf.inputs["Metallic"].default_value = 0.0
        # Add noise texture for wood grain
        tex_coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        noise = nodes.new("ShaderNodeTexNoise")
        color_ramp = nodes.new("ShaderNodeValToRGB")
        mix = nodes.new("ShaderNodeMixRGB")

        noise.inputs["Scale"].default_value = 15.0
        noise.inputs["Detail"].default_value = 8.0
        noise.inputs["Distortion"].default_value = 2.0
        mapping.inputs["Scale"].default_value = (1.0, 8.0, 1.0)

        links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])

        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 0.15
        mix.inputs[1].default_value = hex_to_rgba(hex_color)
        links.new(color_ramp.outputs["Color"], mix.inputs[2])
        links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    else:  # plastic
        bsdf.inputs["Roughness"].default_value = 0.30
        bsdf.inputs["Metallic"].default_value = 0.0

    return mat


def create_table_material() -> bpy.types.Material:
    """Create a wooden table surface material."""
    mat = bpy.data.materials.new(name="table_surface")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = rgb_to_rgba(139, 110, 70)
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Metallic"].default_value = 0.0

    return mat


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

PIECE_NAMES = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King"]
FEN_MAP = {"p": "Pawn", "r": "Rook", "n": "Knight", "b": "Bishop", "q": "Queen", "k": "King"}

# Relative heights for proper Staunton proportions
# These will be scaled to match the board
PIECE_HEIGHTS = {
    "Pawn": 0.55, "Rook": 0.65, "Knight": 0.70,
    "Bishop": 0.80, "Queen": 0.90, "King": 1.0,
}


def load_piece_models(models_dir: str, board_size: float) -> dict:
    """Import STL piece models, scale to board, and return templates.

    The STL models from clarkerubber/Staunton-Pieces have their height
    along the Y axis. We rotate -90° around X to stand them upright (Z-up).
    All pieces are scaled uniformly relative to the King's height so that
    the built-in Staunton proportions are preserved.
    """
    sq_size = board_size / 8.0
    templates = {}

    for name in PIECE_NAMES:
        stl_path = os.path.join(models_dir, f"{name}.STL")
        if not os.path.exists(stl_path):
            print(f"WARNING: Missing model {stl_path}")
            continue

        bpy.ops.wm.stl_import(filepath=stl_path)
        obj = bpy.context.selected_objects[0]
        obj.name = f"template_{name}"

        # Rotate to stand upright: STL has height along Y, Blender uses Z-up
        obj.rotation_euler = (math.radians(90), 0, 0)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(rotation=True)

        templates[name] = obj

    if not templates:
        return {}

    # Scale all uniformly: King height = 1.3 * square size
    king_obj = templates.get("King")
    if king_obj is None:
        king_obj = max(templates.values(), key=lambda o: o.dimensions.z)
    king_height_model = king_obj.dimensions.z
    target_king_height = sq_size * 1.3
    uniform_scale = target_king_height / king_height_model

    for name, obj in templates.items():
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        obj.scale = (uniform_scale, uniform_scale, uniform_scale)
        bpy.ops.object.transform_apply(scale=True)

        # Center XY and align bottom to Z=0
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        local_bb = [mathutils.Vector(v) for v in obj.bound_box]
        min_z = min(v.z for v in local_bb)
        cx = (min(v.x for v in local_bb) + max(v.x for v in local_bb)) / 2
        cy = (min(v.y for v in local_bb) + max(v.y for v in local_bb)) / 2
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(location=True)
        obj.location = (-cx, -cy, -min_z)
        bpy.ops.object.transform_apply(location=True)

        bpy.ops.object.shade_smooth()
        obj.hide_set(True)
        obj.hide_render = True
        obj.select_set(False)
        print(f"  Loaded {name}: height={obj.dimensions.z:.4f}")

    return templates


def create_board(board_size: float, theme: dict) -> list:
    """Create 8x8 chess board from individual square planes."""
    sq_size = board_size / 8.0
    squares = []

    light_mat = create_board_material(
        "board_light", theme["light"], theme.get("texture_type", "vinyl"),
    )
    dark_mat = create_board_material(
        "board_dark", theme["dark"], theme.get("texture_type", "vinyl"),
    )

    for rank in range(8):
        for file in range(8):
            bpy.ops.mesh.primitive_plane_add(
                size=sq_size,
                location=(
                    (file - 3.5) * sq_size,
                    (rank - 3.5) * sq_size,
                    0.0,
                ),
            )
            sq = bpy.context.active_object
            sq.name = f"square_{file}_{rank}"

            is_light = (file + rank) % 2 == 1
            sq.data.materials.append(light_mat if is_light else dark_mat)
            squares.append(sq)

    # Table surface (larger plane underneath)
    bpy.ops.mesh.primitive_plane_add(size=board_size * 3.0, location=(0, 0, -0.002))
    table = bpy.context.active_object
    table.name = "table_surface"
    table.data.materials.append(create_table_material())

    return squares


def setup_lighting(lighting_config: dict):
    """Set up overhead lighting for the scene."""
    color_temp = lighting_config.get("color_temperature", 5200)
    intensity = lighting_config.get("overhead_intensity", 1.0)

    # Convert color temperature to RGB
    temp = color_temp / 100.0
    if temp <= 66:
        r = 1.0
        g = max(0.0, min(1.0, (0.39 * (temp ** 0.5) - 0.63) / 2.5))
    else:
        r = max(0.0, min(1.0, 1.29 * ((temp - 60) ** -0.133)))
        g = max(0.0, min(1.0, 1.13 * ((temp - 60) ** -0.0755)))
    b = 1.0 if temp >= 66 else (
        0.0 if temp <= 19 else max(0.0, min(1.0, (0.54 * ((temp - 10) ** 0.5) - 1.2) / 3.0))
    )

    # Sun light for consistent, uniform illumination
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 3.0))
    sun = bpy.context.active_object
    sun.name = "sun_light"
    sun.data.energy = 3.0 * intensity
    sun.data.color = (r, g, b)
    sun.data.angle = math.radians(15)  # soft shadows
    # Angle: coming from upper-front-left
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(-30))

    # Fill area light from the other side
    bpy.ops.object.light_add(type="AREA", location=(-1.0, -1.5, 1.5))
    fill = bpy.context.active_object
    fill.name = "fill_light"
    fill.data.energy = 100 * intensity
    fill.data.size = 2.5
    fill.data.color = (r, g, b)
    direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector((-1.0, -1.5, 1.5))
    fill.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # Ambient / world lighting
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.05, 0.05, 0.06, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def setup_camera(elevation_deg: float, azimuth_deg: float, board_size: float):
    """Position camera to look at the board from given angle."""
    elev_rad = math.radians(max(elevation_deg, 15.0))
    azim_rad = math.radians(azimuth_deg)

    # Distance from board center — adjusted for 35mm lens
    distance = board_size * 2.0

    # Camera position in spherical coordinates
    x = distance * math.cos(elev_rad) * math.sin(azim_rad)
    y = -distance * math.cos(elev_rad) * math.cos(azim_rad)
    z = distance * math.sin(elev_rad)

    cam_data = bpy.data.cameras.get("Camera") or bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.get("Camera")
    if cam_obj is None:
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)

    cam_obj.location = (x, y, z)

    # Point camera at board center using explicit Euler rotation
    # Camera X rotation = 90° - elevation (0° = looking straight down, 90° = horizontal)
    # Camera Z rotation = azimuth
    cam_obj.rotation_euler = (elev_rad, 0, azim_rad)

    cam_data.lens = 35
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100

    bpy.context.scene.camera = cam_obj
    print(f"  Camera at ({x:.2f}, {y:.2f}, {z:.2f}), elev={elevation_deg:.0f}, azim={azimuth_deg:.0f}")


def setup_render(resolution: int):
    """Configure EEVEE render settings."""
    scene = bpy.context.scene

    # EEVEE engine name varies by Blender version
    if "BLENDER_EEVEE_NEXT" in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    # Use Standard color management (Filmic makes things too dark)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    # EEVEE settings (attributes vary by version, set safely)
    eevee = scene.eevee
    for attr, val in [
        ("taa_render_samples", 48),
        ("use_gtao", True),
        ("use_bloom", False),
        ("use_ssr", True),
        ("use_shadow_high_bitdepth", True),
    ]:
        if hasattr(eevee, attr):
            setattr(eevee, attr, val)


# ---------------------------------------------------------------------------
# FEN parsing and piece placement
# ---------------------------------------------------------------------------


def place_pieces_from_fen(
    fen: str,
    templates: dict,
    white_mat: bpy.types.Material,
    black_mat: bpy.types.Material,
    board_size: float,
    existing_pieces: list,
):
    """Parse FEN and place piece instances on the board.

    Removes any previously placed pieces.
    """
    # Remove existing placed pieces
    for obj in existing_pieces:
        bpy.data.objects.remove(obj, do_unlink=True)
    existing_pieces.clear()

    sq_size = board_size / 8.0

    fen_board = fen.split(" ")[0]
    rank_idx = 7  # FEN starts from rank 8 (index 7)

    for row in fen_board.split("/"):
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                piece_name = FEN_MAP.get(ch.lower())
                is_white = ch.isupper()

                if piece_name and piece_name in templates:
                    template = templates[piece_name]

                    # Duplicate the template (already scaled to board size)
                    new_obj = template.copy()
                    new_obj.data = template.data.copy()
                    bpy.context.scene.collection.objects.link(new_obj)

                    # Position on the board
                    x = (file_idx - 3.5) * sq_size
                    y = (rank_idx - 3.5) * sq_size
                    new_obj.location = (x, y, 0.0)

                    # Rotate knight to face sideways
                    if piece_name == "Knight":
                        facing = math.radians(90) if is_white else math.radians(-90)
                        new_obj.rotation_euler = (0, 0, facing)

                    # Assign material
                    new_obj.data.materials.clear()
                    new_obj.data.materials.append(white_mat if is_white else black_mat)

                    new_obj.hide_set(False)
                    new_obj.hide_render = False
                    new_obj.name = f"piece_{piece_name}_{file_idx}_{rank_idx}"

                    existing_pieces.append(new_obj)

                file_idx += 1
        rank_idx -= 1


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = args.resolution

    # Find models directory (relative to this script)
    script_dir = Path(__file__).parent
    piece_set = manifest.get("piece_set", "staunton")
    models_dir = str(script_dir / "models" / piece_set)

    # --- Scene setup (done once per clip) ---
    clear_scene()
    setup_render(resolution)

    # Board
    board_size = 1.0  # 1 meter board
    theme = manifest["board_theme"]
    create_board(board_size, theme)

    # Load piece models (scaled to board)
    templates = load_piece_models(models_dir, board_size)
    if not templates:
        print("ERROR: No piece models loaded!")
        sys.exit(1)

    # Create piece materials
    mat_info = manifest["material"]
    white_mat = create_piece_material(
        "piece_white", mat_info["white_color"], mat_info["type"],
    )
    black_mat = create_piece_material(
        "piece_black", mat_info["black_color"], mat_info["type"],
    )

    # Lighting
    lighting = manifest.get("lighting", {})
    setup_lighting(lighting)

    # --- Render each frame ---
    placed_pieces: list = []

    for i, frame in enumerate(manifest["frames"]):
        fen = frame["fen"]
        elevation = frame.get("elevation", 45.0)
        azimuth = frame.get("azimuth", 15.0)

        # Place pieces for this position
        place_pieces_from_fen(
            fen, templates, white_mat, black_mat, board_size, placed_pieces,
        )

        # Set camera
        setup_camera(elevation, azimuth, board_size)

        # Render
        out_path = str(output_dir / f"frame_{i:04d}.png")
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered frame {i}: {out_path}")

    print(f"Done. {len(manifest['frames'])} frames rendered to {output_dir}")


if __name__ == "__main__":
    main()
