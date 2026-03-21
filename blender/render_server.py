"""Persistent Blender rendering server for synthetic data generation.

Runs inside Blender as a long-lived TCP server. Accepts rendering
manifests from clients, renders frames, and returns results.
This eliminates the ~2-5 second Blender startup cost per clip.

Start via Makefile:
    make blender-server

Or directly:
    blender --background --python render_server.py -- \
        --resolution 224 --quality training --port 9876
"""

import json
import math
import os
import socket
import struct
import sys
import tempfile
from pathlib import Path

import bpy
import mathutils

# Import shared functions from render_chess.py (same directory)
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from render_chess import (
    BOARD_ELEVATION,
    FEN_MAP,
    PIECE_NAMES,
    _QUALITY_PRESETS,
    _set_bsdf_input,
    create_board,
    create_board_material,
    create_piece_material,
    hex_to_rgba,
    load_piece_models,
    rgb_to_rgba,
    setup_camera,
    setup_lighting,
    setup_render,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--quality", choices=["training", "high"], default="training")
    p.add_argument("--piece-set", default="staunton")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--host", default="127.0.0.1")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Material update helpers (fast recoloring without rebuilding node trees)
# ---------------------------------------------------------------------------


def _find_bsdf(mat):
    """Find the Principled BSDF node in a material."""
    if mat and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                return node
    return None


def recolor_board(theme: dict):
    """Update existing board materials with new colors."""
    light_mat = bpy.data.materials.get("board_light")
    if light_mat:
        bsdf = _find_bsdf(light_mat)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = hex_to_rgba(theme["light"])

    dark_mat = bpy.data.materials.get("board_dark")
    if dark_mat:
        bsdf = _find_bsdf(dark_mat)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = hex_to_rgba(theme["dark"])

    rim_mat = bpy.data.materials.get("rim_mat")
    if rim_mat and "border_color" in theme:
        bsdf = _find_bsdf(rim_mat)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = hex_to_rgba(
                theme["border_color"]
            )

    base_mat = bpy.data.materials.get("board_base_mat")
    if base_mat:
        border = theme.get("border_color", theme["dark"])
        bsdf = _find_bsdf(base_mat)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = hex_to_rgba(border)


def recolor_pieces(mat_info: dict):
    """Update existing piece materials with new colors and properties."""
    mat_type = mat_info["type"]

    for side, color_key in [("white", "white_color"), ("black", "black_color")]:
        mat = bpy.data.materials.get(f"piece_{side}")
        if not mat:
            continue
        bsdf = _find_bsdf(mat)
        if not bsdf:
            continue

        bsdf.inputs["Base Color"].default_value = rgb_to_rgba(*mat_info[color_key])

        if mat_type == "plastic":
            bsdf.inputs["Roughness"].default_value = 0.30
            bsdf.inputs["Metallic"].default_value = 0.0
        elif mat_type == "wood":
            bsdf.inputs["Roughness"].default_value = 0.45
            bsdf.inputs["Metallic"].default_value = 0.0
        elif mat_type == "metal":
            bsdf.inputs["Roughness"].default_value = 0.12
            bsdf.inputs["Metallic"].default_value = 0.95


def update_lighting(lighting_config: dict):
    """Update existing light objects instead of recreating them."""
    color_temp = lighting_config.get("color_temperature", 5200)
    intensity = lighting_config.get("overhead_intensity", 1.0)

    temp = color_temp / 100.0
    if temp <= 66:
        r = 1.0
        g = max(0.0, min(1.0, (0.39 * (temp**0.5) - 0.63) / 2.5))
    else:
        r = max(0.0, min(1.0, 1.29 * ((temp - 60) ** -0.133)))
        g = max(0.0, min(1.0, 1.13 * ((temp - 60) ** -0.0755)))
    b = (
        1.0
        if temp >= 66
        else (
            0.0
            if temp <= 19
            else max(
                0.0, min(1.0, (0.54 * ((temp - 10) ** 0.5) - 1.2) / 3.0)
            )
        )
    )

    sun = bpy.data.objects.get("sun_light")
    if sun:
        sun.data.energy = 5.0 * intensity
        sun.data.color = (r, g, b)

    fill = bpy.data.objects.get("fill_light")
    if fill:
        fill.data.energy = 30 * intensity
        fill.data.color = (r, g, b)

    rim = bpy.data.objects.get("rim_light")
    if rim:
        rim.data.energy = 15 * intensity


# ---------------------------------------------------------------------------
# Diff-based piece placement
# ---------------------------------------------------------------------------


def _parse_fen_to_dict(fen: str) -> dict[tuple[int, int], tuple[str, bool]]:
    """Parse FEN board part into {(file, rank): (piece_name, is_white)}."""
    result = {}
    fen_board = fen.split(" ")[0]
    rank_idx = 7
    for row in fen_board.split("/"):
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                piece_name = FEN_MAP.get(ch.lower())
                if piece_name:
                    result[(file_idx, rank_idx)] = (piece_name, ch.isupper())
                file_idx += 1
        rank_idx -= 1
    return result


def place_pieces_diff(
    fen: str,
    prev_fen: str | None,
    templates: dict,
    white_mat,
    black_mat,
    board_size: float,
    piece_objects: dict[tuple[int, int], "bpy.types.Object"],
):
    """Place pieces using diff against previous FEN for speed."""
    sq_size = board_size / 8.0
    current = _parse_fen_to_dict(fen)

    if prev_fen is not None:
        previous = _parse_fen_to_dict(prev_fen)
    else:
        previous = {}

    # Remove pieces no longer present
    for pos in set(previous.keys()) - set(current.keys()):
        obj = piece_objects.pop(pos, None)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Remove pieces that changed type/color at same square
    for pos in set(previous.keys()) & set(current.keys()):
        if previous[pos] != current[pos]:
            obj = piece_objects.pop(pos, None)
            if obj:
                bpy.data.objects.remove(obj, do_unlink=True)

    # Add new/changed pieces
    for pos, (piece_name, is_white) in current.items():
        if pos in piece_objects:
            continue

        if piece_name not in templates:
            continue

        template = templates[piece_name]
        new_obj = template.copy()
        new_obj.data = template.data.copy()
        bpy.context.scene.collection.objects.link(new_obj)

        file_idx, rank_idx = pos
        x = (file_idx - 3.5) * sq_size
        y = (rank_idx - 3.5) * sq_size
        new_obj.location = (x, y, BOARD_ELEVATION)

        if piece_name == "Knight":
            facing = math.radians(90) if is_white else math.radians(-90)
            new_obj.rotation_euler = (0, 0, facing)

        new_obj.data.materials.clear()
        new_obj.data.materials.append(white_mat if is_white else black_mat)
        new_obj.hide_set(False)
        new_obj.hide_render = False
        new_obj.name = f"piece_{piece_name}_{file_idx}_{rank_idx}"

        piece_objects[pos] = new_obj


# ---------------------------------------------------------------------------
# TCP message protocol (length-prefixed)
# ---------------------------------------------------------------------------


def send_msg(sock: socket.socket, data: bytes) -> None:
    """Send a length-prefixed message."""
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def recv_msg(sock: socket.socket) -> bytes | None:
    """Receive a length-prefixed message. Returns None on disconnect."""
    raw_len = _recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack("!I", raw_len)[0]
    return _recvall(sock, msg_len)


def _recvall(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


# ---------------------------------------------------------------------------
# Server main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    resolution = args.resolution

    models_dir = str(script_dir / "models" / args.piece_set)

    # --- One-time scene setup ---
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)

    setup_render(resolution, quality=args.quality)

    board_size = 1.0
    default_theme = {
        "light": "#F0D9B5",
        "dark": "#B58863",
        "texture_type": "vinyl",
    }
    create_board(board_size, default_theme)

    templates = load_piece_models(models_dir, board_size)
    if not templates:
        print("ERROR: No piece models loaded!", file=sys.stderr)
        sys.exit(1)

    white_mat = create_piece_material("piece_white", [240, 230, 210], "plastic")
    black_mat = create_piece_material("piece_black", [50, 40, 35], "plastic")

    setup_lighting({})

    # Create temp directory for frame output
    tmp_base = tempfile.mkdtemp(prefix="blender_server_")

    # --- Start TCP server ---
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(5)
    print(f"Blender render server listening on {args.host}:{args.port}", file=sys.stderr)

    clip_counter = 0

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"Client connected: {addr}", file=sys.stderr)

            try:
                while True:
                    raw = recv_msg(conn)
                    if raw is None:
                        break

                    try:
                        manifest = json.loads(raw.decode("utf-8"))
                    except json.JSONDecodeError as e:
                        response = {"status": "error", "error": f"Invalid JSON: {e}"}
                        send_msg(conn, json.dumps(response).encode("utf-8"))
                        continue

                    # Create output directory for this clip
                    clip_dir = os.path.join(tmp_base, f"clip_{clip_counter:06d}")
                    os.makedirs(clip_dir, exist_ok=True)
                    clip_counter += 1

                    try:
                        # Recolor for this clip
                        theme = manifest.get("board_theme", default_theme)
                        recolor_board(theme)

                        mat_info = manifest.get("material", {})
                        if mat_info:
                            recolor_pieces(mat_info)

                        lighting = manifest.get("lighting", {})
                        if lighting:
                            update_lighting(lighting)

                        # Render frames
                        frames_out = []
                        piece_objects = {}
                        prev_fen = None

                        for i, frame in enumerate(manifest.get("frames", [])):
                            fen = frame["fen"]
                            elevation = frame.get("elevation", 45.0)
                            azimuth = frame.get("azimuth", 15.0)

                            place_pieces_diff(
                                fen, prev_fen, templates,
                                white_mat, black_mat, board_size,
                                piece_objects,
                            )
                            prev_fen = fen

                            setup_camera(elevation, azimuth, board_size)

                            out_path = os.path.join(clip_dir, f"frame_{i:04d}.png")
                            bpy.context.scene.render.filepath = out_path
                            bpy.ops.render.render(write_still=True)
                            frames_out.append(out_path)

                        # Clean up placed pieces
                        for obj in piece_objects.values():
                            bpy.data.objects.remove(obj, do_unlink=True)
                        piece_objects.clear()

                        response = {"status": "ok", "frames": frames_out}
                        send_msg(conn, json.dumps(response).encode("utf-8"))

                    except Exception as e:
                        for obj in piece_objects.values():
                            try:
                                bpy.data.objects.remove(obj, do_unlink=True)
                            except Exception:
                                pass
                        response = {"status": "error", "error": str(e)}
                        send_msg(conn, json.dumps(response).encode("utf-8"))

            finally:
                conn.close()
                print(f"Client disconnected: {addr}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nBlender render server shutting down.", file=sys.stderr)
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
