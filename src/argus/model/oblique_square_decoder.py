"""Geometry-aware learned square queries for full-board crops."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

_NUM_SQUARES = 64
_NUM_GRID_POINTS = 81


class ObliqueSquareQueryDecoder(nn.Module):
    """Decode 64 square tokens jointly from dense board patch tokens.

    The decoder seeds each square with a learned query plus a geometry embedding
    derived from the board corners, then cross-attends into the full patch grid.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(embed_dim * mlp_ratio), embed_dim)
        self.square_queries = nn.Parameter(torch.zeros(_NUM_SQUARES, embed_dim))
        self.patch_position_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.square_geometry_proj = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.register_buffer(
            "canonical_square_coords",
            _build_canonical_square_coordinates(),
            persistent=False,
        )
        self.register_buffer(
            "canonical_grid_points",
            _build_canonical_grid_points(),
            persistent=False,
        )
        nn.init.normal_(self.square_queries, std=0.02)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        corners: torch.Tensor | None = None,
        image_size: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        spatial_tokens = _extract_spatial_tokens(patch_tokens)
        grid_size = _infer_grid_size(spatial_tokens.shape[1])
        batch_size = spatial_tokens.shape[0]
        image_height, image_width = _resolve_image_size(image_size)

        patch_positions = _build_patch_positions(
            grid_size,
            batch_size=batch_size,
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        patch_features = spatial_tokens + self.patch_position_proj(patch_positions)

        square_geometry = _build_square_geometry(
            corners=corners,
            canonical_grid_points=cast(torch.Tensor, self.canonical_grid_points),
            canonical_square_coords=cast(torch.Tensor, self.canonical_square_coords),
            image_height=image_height,
            image_width=image_width,
            batch_size=batch_size,
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        queries = self.square_queries.unsqueeze(0) + self.square_geometry_proj(square_geometry)

        attended, _ = self.cross_attn(
            self.query_norm(queries),
            self.patch_norm(patch_features),
            self.patch_norm(patch_features),
            need_weights=False,
        )
        square_tokens = queries + attended
        square_tokens = square_tokens + self.ffn(self.ffn_norm(square_tokens))
        result: torch.Tensor = self.output_norm(square_tokens)
        return result


def _extract_spatial_tokens(patch_tokens: torch.Tensor) -> torch.Tensor:
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch tokens shaped (B, N, D), got {tuple(patch_tokens.shape)}")

    token_count = patch_tokens.shape[1]
    cls_grid_size = int((token_count - 1) ** 0.5)
    if cls_grid_size * cls_grid_size == token_count - 1:
        return patch_tokens[:, 1:, :]

    plain_grid_size = int(token_count**0.5)
    if plain_grid_size * plain_grid_size == token_count:
        return patch_tokens

    raise ValueError(
        f"Token count {token_count} is neither a square grid nor a square grid plus CLS"
    )


def _infer_grid_size(token_count: int) -> int:
    grid_size = int(token_count**0.5)
    if grid_size * grid_size != token_count:
        raise ValueError(f"Expected square patch grid, got {token_count} tokens")
    return grid_size


def _resolve_image_size(image_size: int | tuple[int, int] | None) -> tuple[int, int]:
    if image_size is None:
        raise ValueError("image_size is required for geometry-aware square decoding")
    if isinstance(image_size, int):
        return image_size, image_size
    return int(image_size[0]), int(image_size[1])


def _build_patch_positions(
    grid_size: int,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axis = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    positions = torch.stack((xx, yy), dim=-1).reshape(1, grid_size * grid_size, 2)
    return positions.expand(batch_size, -1, -1)


def _build_square_geometry(
    *,
    corners: torch.Tensor | None,
    canonical_grid_points: torch.Tensor,
    canonical_square_coords: torch.Tensor,
    image_height: int,
    image_width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    corners_tensor = _resolve_corners(
        corners,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
        device=device,
    )
    norm_corners = corners_tensor.clone()
    norm_corners[..., 0] /= max(float(image_width - 1), 1.0)
    norm_corners[..., 1] /= max(float(image_height - 1), 1.0)

    homographies = _unit_square_to_quad_homography(norm_corners)
    grid_points = _project_points(homographies, canonical_grid_points.to(device=device))
    grid = grid_points.reshape(batch_size, 9, 9, 2)

    top_left = grid[:, :-1, :-1, :]
    top_right = grid[:, :-1, 1:, :]
    bottom_right = grid[:, 1:, 1:, :]
    bottom_left = grid[:, 1:, :-1, :]

    center = (top_left + top_right + bottom_right + bottom_left) / 4.0
    width = (top_right - top_left).norm(dim=-1) + (bottom_right - bottom_left).norm(dim=-1)
    height = (bottom_left - top_left).norm(dim=-1) + (bottom_right - top_right).norm(dim=-1)
    width = (width / 2.0).reshape(batch_size, _NUM_SQUARES, 1)
    height = (height / 2.0).reshape(batch_size, _NUM_SQUARES, 1)
    center = center.reshape(batch_size, _NUM_SQUARES, 2)

    canonical = canonical_square_coords.to(device=device, dtype=torch.float32).unsqueeze(0)
    features = torch.cat((center, width, height, canonical.expand(batch_size, -1, -1)), dim=-1)
    return features.to(dtype=dtype)


def _resolve_corners(
    corners: torch.Tensor | None,
    *,
    batch_size: int,
    image_height: int,
    image_width: int,
    device: torch.device,
) -> torch.Tensor:
    if corners is None:
        canonical = torch.tensor(
            [
                [0.0, 0.0],
                [float(image_width - 1), 0.0],
                [float(image_width - 1), float(image_height - 1)],
                [0.0, float(image_height - 1)],
            ],
            dtype=torch.float32,
            device=device,
        )
        return canonical.unsqueeze(0).expand(batch_size, -1, -1)
    if corners.shape != (batch_size, 4, 2):
        raise ValueError(
            f"Expected corners with shape ({batch_size}, 4, 2), got {tuple(corners.shape)}"
        )
    return corners.to(device=device, dtype=torch.float32)


def _unit_square_to_quad_homography(corners: torch.Tensor) -> torch.Tensor:
    source = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
        device=corners.device,
    ).unsqueeze(0)
    batch_size = corners.shape[0]
    a = torch.zeros((batch_size, 8, 8), dtype=torch.float32, device=corners.device)
    b = torch.zeros((batch_size, 8), dtype=torch.float32, device=corners.device)

    for point_index in range(4):
        u = source[:, point_index, 0]
        v = source[:, point_index, 1]
        x = corners[:, point_index, 0]
        y = corners[:, point_index, 1]
        row = point_index * 2

        a[:, row, 0] = u
        a[:, row, 1] = v
        a[:, row, 2] = 1.0
        a[:, row, 6] = -x * u
        a[:, row, 7] = -x * v
        b[:, row] = x

        a[:, row + 1, 3] = u
        a[:, row + 1, 4] = v
        a[:, row + 1, 5] = 1.0
        a[:, row + 1, 6] = -y * u
        a[:, row + 1, 7] = -y * v
        b[:, row + 1] = y

    solution = torch.linalg.solve(a, b)
    ones = torch.ones((batch_size, 1), dtype=torch.float32, device=corners.device)
    return torch.cat((solution, ones), dim=1).reshape(batch_size, 3, 3)


def _project_points(homographies: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    batch_size = homographies.shape[0]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points shaped (N, 2), got {tuple(points.shape)}")
    homogeneous = torch.cat(
        (
            points.unsqueeze(0).expand(batch_size, -1, -1),
            torch.ones((batch_size, points.shape[0], 1), dtype=points.dtype, device=points.device),
        ),
        dim=-1,
    )
    projected = homogeneous @ homographies.transpose(1, 2)
    denom = projected[..., 2:]
    denom = torch.where(denom >= 0, denom.clamp_min(1e-6), denom.clamp_max(-1e-6))
    return projected[..., :2] / denom


def _build_canonical_square_coordinates() -> torch.Tensor:
    coords = [((col + 0.5) / 8.0, (row + 0.5) / 8.0) for row in range(8) for col in range(8)]
    return torch.tensor(coords, dtype=torch.float32)


def _build_canonical_grid_points() -> torch.Tensor:
    points = [(col / 8.0, row / 8.0) for row in range(9) for col in range(9)]
    return torch.tensor(points, dtype=torch.float32)
