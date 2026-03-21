"""Procedural 3D chess piece renderer using surface-of-revolution geometry.

Renders realistic Staunton chess pieces from actual 3D geometry with Phong
shading. Most pieces (Pawn, Rook, Bishop, Queen, King) are rotationally
symmetric — their shape is a 2D profile curve revolved around the vertical
axis. The Knight uses a procedural heightmap approach.

All rendering is pure numpy (no Blender or external 3D engine). Piece
sprites are cached per (type, color, material, size) for fast per-frame
compositing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import chess
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

from argus.datagen.board_themes import _perturb_color


# ---------------------------------------------------------------------------
# Profile curves — real Staunton proportions (King height = 1.0)
# ---------------------------------------------------------------------------


@dataclass
class PieceProfile:
    """Cross-section profile for a surface-of-revolution piece."""

    name: str
    control_points: list[tuple[float, float]]  # [(height, radius), ...]
    height: float  # total height in normalized units

    _interp: Any = field(default=None, init=False, repr=False)

    def _ensure_interp(self) -> None:
        if self._interp is None:
            ys = [p[0] for p in self.control_points]
            rs = [p[1] for p in self.control_points]
            self._interp = PchipInterpolator(ys, rs)

    def radius_at(self, y: np.ndarray | float) -> np.ndarray:
        """Interpolated radius at height(s) y. Clamps to 0 outside range."""
        self._ensure_interp()
        y = np.asarray(y, dtype=np.float64)
        r = self._interp(np.clip(y, self.control_points[0][0], self.control_points[-1][0]))
        return np.maximum(r, 0.0)

    def radius_derivative(self, y: np.ndarray | float) -> np.ndarray:
        """dr/dy at height(s) y via the interpolator derivative."""
        self._ensure_interp()
        y = np.asarray(y, dtype=np.float64)
        return self._interp(np.clip(y, self.control_points[0][0], self.control_points[-1][0]), 1)


PAWN_PROFILE = PieceProfile(
    name="pawn",
    height=0.55,
    control_points=[
        (0.000, 0.00),
        (0.005, 0.38),
        (0.030, 0.38),
        (0.050, 0.36),
        (0.080, 0.32),
        (0.100, 0.28),
        (0.110, 0.30),
        (0.120, 0.28),
        (0.160, 0.14),
        (0.250, 0.11),
        (0.320, 0.10),
        (0.360, 0.12),
        (0.380, 0.16),
        (0.420, 0.20),
        (0.460, 0.20),
        (0.500, 0.17),
        (0.530, 0.10),
        (0.545, 0.04),
        (0.550, 0.00),
    ],
)

ROOK_PROFILE = PieceProfile(
    name="rook",
    height=0.65,
    control_points=[
        (0.000, 0.00),
        (0.005, 0.38),
        (0.030, 0.38),
        (0.050, 0.36),
        (0.080, 0.32),
        (0.100, 0.28),
        (0.110, 0.30),
        (0.120, 0.28),
        (0.160, 0.15),
        (0.220, 0.13),
        (0.300, 0.12),
        (0.380, 0.14),
        (0.440, 0.17),
        (0.480, 0.20),
        (0.520, 0.24),
        (0.540, 0.26),
        (0.560, 0.26),
        (0.580, 0.24),
        (0.610, 0.24),
        (0.640, 0.22),
        (0.650, 0.00),
    ],
)

KNIGHT_PROFILE = PieceProfile(
    name="knight",
    height=0.70,
    control_points=[
        (0.000, 0.00),
        (0.005, 0.38),
        (0.030, 0.38),
        (0.050, 0.36),
        (0.080, 0.32),
        (0.100, 0.28),
        (0.110, 0.30),
        (0.120, 0.28),
        (0.160, 0.14),
        (0.200, 0.12),
        # Horse head region — profile just defines the base
        (0.250, 0.10),
        (0.700, 0.00),
    ],
)

BISHOP_PROFILE = PieceProfile(
    name="bishop",
    height=0.80,
    control_points=[
        (0.000, 0.00),
        (0.005, 0.38),
        (0.030, 0.38),
        (0.050, 0.36),
        (0.080, 0.32),
        (0.100, 0.28),
        (0.110, 0.30),
        (0.120, 0.28),
        (0.160, 0.14),
        (0.250, 0.11),
        (0.340, 0.10),
        (0.370, 0.12),
        (0.390, 0.14),
        (0.400, 0.13),
        (0.440, 0.16),
        (0.500, 0.19),
        (0.560, 0.20),
        (0.620, 0.19),
        (0.680, 0.15),
        (0.720, 0.10),
        (0.750, 0.06),
        (0.760, 0.04),
        (0.770, 0.05),
        (0.780, 0.05),
        (0.790, 0.03),
        (0.800, 0.00),
    ],
)

QUEEN_PROFILE = PieceProfile(
    name="queen",
    height=0.90,
    control_points=[
        (0.000, 0.00),
        (0.005, 0.38),
        (0.030, 0.38),
        (0.050, 0.36),
        (0.080, 0.32),
        (0.100, 0.28),
        (0.110, 0.30),
        (0.120, 0.28),
        (0.160, 0.14),
        (0.260, 0.11),
        (0.360, 0.10),
        (0.390, 0.12),
        (0.410, 0.14),
        (0.420, 0.13),
        (0.460, 0.17),
        (0.520, 0.21),
        (0.580, 0.23),
        (0.640, 0.22),
        (0.700, 0.18),
        (0.740, 0.15),
        (0.770, 0.11),
        (0.790, 0.07),
        (0.800, 0.04),
        (0.820, 0.06),
        (0.850, 0.07),
        (0.870, 0.06),
        (0.890, 0.03),
        (0.900, 0.00),
    ],
)

KING_PROFILE = PieceProfile(
    name="king",
    height=1.0,
    control_points=[
        (0.000, 0.00),
        (0.005, 0.40),
        (0.030, 0.40),
        (0.050, 0.38),
        (0.080, 0.34),
        (0.100, 0.30),
        (0.110, 0.32),
        (0.120, 0.30),
        (0.160, 0.15),
        (0.280, 0.12),
        (0.380, 0.11),
        (0.410, 0.13),
        (0.430, 0.15),
        (0.440, 0.14),
        (0.480, 0.18),
        (0.540, 0.22),
        (0.600, 0.24),
        (0.660, 0.23),
        (0.720, 0.19),
        (0.760, 0.16),
        (0.780, 0.18),
        (0.790, 0.16),
        (0.820, 0.10),
        (0.850, 0.06),
        (0.860, 0.04),
        (0.870, 0.04),
        (0.920, 0.04),
        (0.950, 0.04),
        (0.990, 0.02),
        (1.000, 0.00),
    ],
)

PROFILES: dict[int, PieceProfile] = {
    chess.PAWN: PAWN_PROFILE,
    chess.ROOK: ROOK_PROFILE,
    chess.KNIGHT: KNIGHT_PROFILE,
    chess.BISHOP: BISHOP_PROFILE,
    chess.QUEEN: QUEEN_PROFILE,
    chess.KING: KING_PROFILE,
}


# ---------------------------------------------------------------------------
# Materials — Phong shading parameters
# ---------------------------------------------------------------------------


@dataclass
class PieceMaterial:
    """Phong/Blinn material for 3D piece rendering."""

    name: str
    material_type: str  # "plastic", "wood", "metal"

    ambient: float
    diffuse: float
    specular: float
    shininess: float

    white_color: tuple[int, int, int]
    black_color: tuple[int, int, int]

    surface_noise: float
    grain_intensity: float

    weight: float = 1.0

    def with_perturbation(self, rng: random.Random) -> PieceMaterial:
        """Return a copy with small random variation."""
        wc = _perturb_color(self.white_color, rng)
        bc = _perturb_color(self.black_color, rng)
        j = lambda v: v * rng.uniform(0.90, 1.10)  # noqa: E731
        return PieceMaterial(
            name=self.name,
            material_type=self.material_type,
            ambient=min(1.0, max(0.05, j(self.ambient))),
            diffuse=min(1.0, max(0.1, j(self.diffuse))),
            specular=min(1.0, max(0.0, j(self.specular))),
            shininess=max(5.0, j(self.shininess)),
            white_color=wc,
            black_color=bc,
            surface_noise=j(self.surface_noise),
            grain_intensity=j(self.grain_intensity),
            weight=self.weight,
        )


PIECE_MATERIALS: list[PieceMaterial] = [
    PieceMaterial(
        name="tournament_plastic",
        material_type="plastic",
        ambient=0.25, diffuse=0.65, specular=0.35, shininess=30.0,
        white_color=(235, 225, 210),
        black_color=(55, 45, 35),
        surface_noise=3.0, grain_intensity=0.0,
        weight=0.30,
    ),
    PieceMaterial(
        name="club_plastic",
        material_type="plastic",
        ambient=0.25, diffuse=0.60, specular=0.40, shininess=35.0,
        white_color=(245, 240, 230),
        black_color=(40, 35, 30),
        surface_noise=2.5, grain_intensity=0.0,
        weight=0.20,
    ),
    PieceMaterial(
        name="basic_plastic",
        material_type="plastic",
        ambient=0.28, diffuse=0.60, specular=0.30, shininess=25.0,
        white_color=(250, 250, 245),
        black_color=(35, 35, 35),
        surface_noise=2.0, grain_intensity=0.0,
        weight=0.15,
    ),
    PieceMaterial(
        name="boxwood_ebony",
        material_type="wood",
        ambient=0.22, diffuse=0.70, specular=0.15, shininess=12.0,
        white_color=(220, 195, 150),
        black_color=(70, 45, 25),
        surface_noise=4.0, grain_intensity=0.6,
        weight=0.12,
    ),
    PieceMaterial(
        name="sheesham_boxwood",
        material_type="wood",
        ambient=0.22, diffuse=0.70, specular=0.12, shininess=10.0,
        white_color=(225, 200, 160),
        black_color=(100, 60, 30),
        surface_noise=4.0, grain_intensity=0.5,
        weight=0.08,
    ),
    PieceMaterial(
        name="brushed_steel",
        material_type="metal",
        ambient=0.18, diffuse=0.45, specular=0.70, shininess=100.0,
        white_color=(190, 190, 195),
        black_color=(60, 60, 65),
        surface_noise=2.0, grain_intensity=0.0,
        weight=0.05,
    ),
    PieceMaterial(
        name="brass_pewter",
        material_type="metal",
        ambient=0.18, diffuse=0.50, specular=0.65, shininess=80.0,
        white_color=(200, 185, 140),
        black_color=(75, 70, 65),
        surface_noise=2.5, grain_intensity=0.0,
        weight=0.05,
    ),
    PieceMaterial(
        name="weighted_plastic",
        material_type="plastic",
        ambient=0.24, diffuse=0.65, specular=0.32, shininess=28.0,
        white_color=(230, 215, 195),
        black_color=(60, 50, 40),
        surface_noise=3.5, grain_intensity=0.0,
        weight=0.05,
    ),
]


def select_random_material(rng: random.Random) -> PieceMaterial:
    """Select a piece material using weighted random sampling."""
    weights = [m.weight for m in PIECE_MATERIALS]
    total = sum(weights)
    r = rng.uniform(0, total)
    cumulative = 0.0
    for mat in PIECE_MATERIALS:
        cumulative += mat.weight
        if r <= cumulative:
            return mat
    return PIECE_MATERIALS[-1]


# ---------------------------------------------------------------------------
# 3D rendering engine
# ---------------------------------------------------------------------------


def _phong_shade(
    normals: np.ndarray,
    mask: np.ndarray,
    base_color: np.ndarray,
    material: PieceMaterial,
    light_dir: np.ndarray,
) -> np.ndarray:
    """Apply Phong shading to a surface given normals and material.

    Args:
        normals: (H, W, 3) float64 surface normal vectors (normalized).
        mask: (H, W) bool — which pixels to shade.
        base_color: (3,) float64 base color [0, 255].
        material: PieceMaterial with Phong coefficients.
        light_dir: (3,) float64 normalized light direction (toward light).

    Returns:
        (H, W, 3) float64 RGB values [0, 255].
    """
    h, w = mask.shape
    result = np.zeros((h, w, 3), dtype=np.float64)
    if not np.any(mask):
        return result

    n = normals[mask]  # (N, 3)
    l = light_dir  # (3,)

    # Ambient
    ambient = material.ambient * base_color

    # Diffuse: max(N·L, 0)
    n_dot_l = np.clip(np.sum(n * l, axis=1), 0.0, 1.0)  # (N,)
    diffuse = material.diffuse * n_dot_l[:, None] * base_color  # (N, 3)

    # Specular: (R·V)^shininess where V = (0, 0, 1) and R = 2(N·L)N - L
    v = np.array([0.0, 0.0, 1.0])
    r_vec = 2.0 * n_dot_l[:, None] * n - l  # (N, 3)
    r_dot_v = np.clip(np.sum(r_vec * v, axis=1), 0.0, 1.0)  # (N,)
    spec_color = np.array([255.0, 255.0, 255.0])
    if material.material_type == "metal":
        # Metallic specular is tinted toward base color
        spec_color = base_color * 0.5 + 128.0
    specular = material.specular * (r_dot_v ** material.shininess)[:, None] * spec_color  # (N, 3)

    result[mask] = ambient + diffuse + specular
    return result


def render_revolution_piece(
    profile: PieceProfile,
    material: PieceMaterial,
    is_white: bool,
    sq_size: int,
    light_dir: tuple[float, float, float] = (-0.4, 0.3, 0.86),
    rng: random.Random | None = None,
) -> Image.Image:
    """Render a surface-of-revolution piece as an RGBA sprite.

    Args:
        profile: PieceProfile defining the piece shape.
        material: PieceMaterial for Phong shading.
        is_white: True for white pieces, False for black.
        sq_size: Output image size (square).
        light_dir: (x, y, z) light direction. Default: from upper-left.
        rng: Random for surface texture.

    Returns:
        RGBA PIL Image of size (sq_size, sq_size).
    """
    base_color = np.array(
        material.white_color if is_white else material.black_color, dtype=np.float64
    )
    light = np.array(light_dir, dtype=np.float64)
    light /= np.linalg.norm(light) + 1e-12

    # Coordinate system: piece centered in image
    # Piece fills ~80% of width, bottom-aligned with ~5% margin
    margin_bottom = int(sq_size * 0.05)
    max_radius = max(p[1] for p in profile.control_points)
    piece_width_px = sq_size * 0.80
    scale = piece_width_px / (2.0 * max_radius) if max_radius > 0 else 1.0
    piece_height_px = profile.height * scale

    # Map pixel coords to world coords
    cx = sq_size / 2.0
    py = np.arange(sq_size, dtype=np.float64)
    px = np.arange(sq_size, dtype=np.float64)
    px_grid, py_grid = np.meshgrid(px, py)

    # World x: horizontal offset from center
    x_world = (px_grid - cx) / scale
    # World y: height from base (bottom of image = 0)
    y_world = (sq_size - 1 - py_grid - margin_bottom) / scale

    # Evaluate profile radius at each height
    y_flat = y_world.ravel()
    valid_y = (y_flat >= 0) & (y_flat <= profile.height)
    r_flat = np.zeros_like(y_flat)
    if np.any(valid_y):
        r_flat[valid_y] = profile.radius_at(y_flat[valid_y])
    r_grid = r_flat.reshape(sq_size, sq_size)

    # Silhouette mask
    x_abs = np.abs(x_world)
    mask = (x_abs <= r_grid) & (y_world >= 0) & (y_world <= profile.height) & (r_grid > 0.001)

    # Anti-aliasing: sub-pixel coverage at edges
    edge_dist = r_grid - x_abs
    aa_width = 1.5 / scale  # 1.5 pixels in world space
    coverage = np.clip(edge_dist / aa_width, 0.0, 1.0)
    # Also anti-alias top and bottom
    coverage *= np.clip(y_world / aa_width, 0.0, 1.0)
    coverage *= np.clip((profile.height - y_world) / aa_width, 0.0, 1.0)

    # Surface normals for the surface of revolution
    normals = np.zeros((sq_size, sq_size, 3), dtype=np.float64)
    if np.any(mask):
        r_at = r_grid[mask]  # radius at this height
        x_at = x_world[mask]  # horizontal position
        y_at = y_world[mask]  # height

        # Azimuthal angle (from front)
        cos_phi = np.clip(x_at / (r_at + 1e-12), -1.0, 1.0)
        sin_phi = np.sqrt(np.clip(1.0 - cos_phi ** 2, 0.0, 1.0))

        # Profile derivative dr/dy
        dr_dy = profile.radius_derivative(y_at)

        # Normal components
        norm_factor = np.sqrt(1.0 + dr_dy ** 2)
        nx = cos_phi / norm_factor
        ny = -dr_dy / norm_factor
        nz = sin_phi / norm_factor

        normals[mask, 0] = nx
        normals[mask, 1] = ny
        normals[mask, 2] = nz

    # Phong shading
    rgb = _phong_shade(normals, mask, base_color, material, light)

    # Surface texture
    if rng is not None and material.surface_noise > 0.5:
        rs = np.random.RandomState(rng.randint(0, 2**31))
        if material.material_type == "wood" and material.grain_intensity > 0:
            noise = rs.normal(0, 1, (sq_size, sq_size))
            grain = gaussian_filter(noise, sigma=(0.5, sq_size * 0.12))
            grain = grain / (np.abs(grain).max() + 1e-8)
            grain *= material.grain_intensity * 12.0
            fine = rs.normal(0, material.surface_noise, (sq_size, sq_size))
            for c in range(3):
                rgb[:, :, c][mask] += (grain[mask] + fine[mask])
        elif material.material_type == "metal":
            noise = rs.normal(0, 1, (sq_size, sq_size))
            brushed = gaussian_filter(noise, sigma=(0.3, sq_size * 0.08))
            brushed = brushed / (np.abs(brushed).max() + 1e-8)
            brushed *= material.surface_noise * 1.5
            for c in range(3):
                rgb[:, :, c][mask] += brushed[mask]
        else:
            noise = rs.normal(0, material.surface_noise, (sq_size, sq_size))
            for c in range(3):
                rgb[:, :, c][mask] += noise[mask]

    # Assemble RGBA
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.zeros((sq_size, sq_size), dtype=np.uint8)
    alpha[mask] = np.clip(coverage[mask] * 255, 0, 255).astype(np.uint8)

    result = np.zeros((sq_size, sq_size, 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = alpha
    return Image.fromarray(result, "RGBA")


def _build_knight_heightmap(sq_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a procedural heightmap and alpha mask for the knight.

    The knight is constructed from overlapping ellipsoidal blobs:
    base cylinder + neck + head + muzzle + ears.

    Returns:
        (heightmap, alpha_mask) — both (sq_size, sq_size) float64.
        heightmap: depth values (higher = closer to camera), [0, 1].
        alpha_mask: [0, 1] coverage.
    """
    py = np.arange(sq_size, dtype=np.float64)
    px = np.arange(sq_size, dtype=np.float64)
    px_grid, py_grid = np.meshgrid(px, py)

    # Normalize to [0, 1] coords
    x = px_grid / sq_size
    y = 1.0 - py_grid / sq_size  # y=0 at bottom, y=1 at top

    heightmap = np.zeros((sq_size, sq_size), dtype=np.float64)

    def add_ellipsoid(
        cx: float, cy: float, rx: float, ry: float,
        depth: float, rotation: float = 0.0,
    ) -> None:
        """Add an ellipsoidal blob to the heightmap."""
        dx = x - cx
        dy = y - cy
        if rotation != 0.0:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            rdx = dx * cos_r + dy * sin_r
            rdy = -dx * sin_r + dy * cos_r
            dx, dy = rdx, rdy
        dist_sq = (dx / rx) ** 2 + (dy / ry) ** 2
        blob = depth * np.clip(1.0 - dist_sq, 0.0, 1.0)
        np.maximum(heightmap, blob, out=heightmap)

    # Base (shared Staunton base — wide, low)
    add_ellipsoid(0.50, 0.08, 0.38, 0.08, 0.5)
    add_ellipsoid(0.50, 0.15, 0.28, 0.06, 0.55)
    # Collar rim
    add_ellipsoid(0.50, 0.18, 0.30, 0.03, 0.58)

    # Neck (tilted cylinder rising from base)
    add_ellipsoid(0.46, 0.30, 0.14, 0.12, 0.65, rotation=0.15)
    add_ellipsoid(0.44, 0.42, 0.13, 0.10, 0.70, rotation=0.2)

    # Head (main mass — egg shape, tilted forward)
    add_ellipsoid(0.42, 0.55, 0.15, 0.14, 0.80, rotation=0.25)
    add_ellipsoid(0.40, 0.62, 0.14, 0.10, 0.82, rotation=0.3)

    # Muzzle (protruding forward and down)
    add_ellipsoid(0.35, 0.52, 0.10, 0.06, 0.75, rotation=0.4)

    # Forehead bulge
    add_ellipsoid(0.43, 0.68, 0.10, 0.06, 0.78)

    # Ears (two small bumps at top)
    add_ellipsoid(0.38, 0.74, 0.04, 0.04, 0.70)
    add_ellipsoid(0.46, 0.76, 0.04, 0.04, 0.68)

    # Mane ridge (back of neck/head)
    add_ellipsoid(0.52, 0.50, 0.06, 0.18, 0.60, rotation=-0.1)

    # Smooth the heightmap for natural curves
    heightmap = gaussian_filter(heightmap, sigma=sq_size * 0.02)

    # Alpha mask from heightmap (anything with depth > threshold)
    alpha = np.clip(heightmap * 8.0, 0.0, 1.0)
    # Soften edges
    alpha = gaussian_filter(alpha, sigma=0.8)
    alpha = np.clip(alpha, 0.0, 1.0)

    return heightmap, alpha


def render_knight_sprite(
    material: PieceMaterial,
    is_white: bool,
    sq_size: int,
    light_dir: tuple[float, float, float] = (-0.4, 0.3, 0.86),
    rng: random.Random | None = None,
) -> Image.Image:
    """Render the knight piece using a procedural heightmap.

    The knight is not a surface of revolution, so we use a depth map
    approach: define the piece shape as a heightmap, derive normals
    from the gradient, and apply the same Phong shading.
    """
    base_color = np.array(
        material.white_color if is_white else material.black_color, dtype=np.float64
    )
    light = np.array(light_dir, dtype=np.float64)
    light /= np.linalg.norm(light) + 1e-12

    heightmap, alpha_raw = _build_knight_heightmap(sq_size)
    mask = alpha_raw > 0.05

    # Compute normals from heightmap gradient
    # dz/dx and dz/dy give the surface slope
    dz_dx = np.gradient(heightmap, axis=1)
    dz_dy = -np.gradient(heightmap, axis=0)  # negate because y is inverted

    normals = np.zeros((sq_size, sq_size, 3), dtype=np.float64)
    if np.any(mask):
        nx = -dz_dx[mask]
        ny = -dz_dy[mask]
        nz = np.ones(np.sum(mask), dtype=np.float64) * 0.3

        length = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
        normals[mask, 0] = nx / length
        normals[mask, 1] = ny / length
        normals[mask, 2] = nz / length

    # Phong shading
    rgb = _phong_shade(normals, mask, base_color, material, light)

    # Surface texture
    if rng is not None and material.surface_noise > 0.5:
        rs = np.random.RandomState(rng.randint(0, 2**31))
        noise = rs.normal(0, material.surface_noise, (sq_size, sq_size))
        for c in range(3):
            rgb[:, :, c][mask] += noise[mask]

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.clip(alpha_raw * 255, 0, 255).astype(np.uint8)

    result = np.zeros((sq_size, sq_size, 4), dtype=np.uint8)
    result[:, :, :3] = rgb
    result[:, :, 3] = alpha
    return Image.fromarray(result, "RGBA")


def render_piece_sprite(
    piece_type: int,
    is_white: bool,
    material: PieceMaterial,
    sq_size: int,
    light_dir: tuple[float, float, float] = (-0.4, 0.3, 0.86),
    rng: random.Random | None = None,
) -> Image.Image:
    """Render a single chess piece as an RGBA sprite.

    Public entry point. Dispatches to revolution or knight renderer.
    """
    if piece_type == chess.KNIGHT:
        return render_knight_sprite(material, is_white, sq_size, light_dir, rng)

    profile = PROFILES[piece_type]
    return render_revolution_piece(profile, material, is_white, sq_size, light_dir, rng)


# ---------------------------------------------------------------------------
# Cache and board compositing
# ---------------------------------------------------------------------------


class PieceRenderCache:
    """Cache for rendered piece sprites, keyed by (type, color, material, size)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, bool, str, int], Image.Image] = {}

    def get_or_render(
        self,
        piece_type: int,
        is_white: bool,
        material: PieceMaterial,
        sq_size: int,
        light_dir: tuple[float, float, float],
        rng: random.Random | None,
    ) -> Image.Image:
        key = (piece_type, is_white, material.name, sq_size)
        if key not in self._cache:
            self._cache[key] = render_piece_sprite(
                piece_type, is_white, material, sq_size, light_dir, rng,
            )
        return self._cache[key]


def render_pieces_layer(
    board: chess.Board,
    size: int,
    flipped: bool,
    material: PieceMaterial,
    light_dir: tuple[float, float, float] = (-0.4, 0.3, 0.86),
    cache: PieceRenderCache | None = None,
    rng: random.Random | None = None,
) -> Image.Image:
    """Render all pieces on the board as a single RGBA layer.

    Each piece is rendered as a 3D sprite and placed at its board square.
    Returns an RGBA image the same size as the board.
    """
    if cache is None:
        cache = PieceRenderCache()

    sq_size = size // 8
    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    for rank in range(8):
        for file in range(8):
            display_rank = 7 - rank if not flipped else rank
            display_file = file if not flipped else 7 - file
            sq = chess.square(display_file, display_rank)
            piece = board.piece_at(sq)
            if piece is None:
                continue

            sprite = cache.get_or_render(
                piece.piece_type,
                piece.color == chess.WHITE,
                material,
                sq_size,
                light_dir,
                rng,
            )
            x0 = file * sq_size
            y0 = rank * sq_size
            result.paste(sprite, (x0, y0), sprite)

    return result
