"""Strategy 5: Intra-frame visual clustering.

Group the 64 squares by visual similarity. Same piece type on same
square color should look nearly identical. Assign cluster labels
using chess constraints (piece counts, position rules).
"""

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from shared import (
    _inner_crop,
    build_board,
    crop_squares,
    detect_orientation,
    detect_theme,
    find_board_in_frame,
)

STRATEGY_NAME = "S5: Clustering"

CANONICAL = 24  # Resize squares to this for comparison


def _square_feature(square: np.ndarray) -> np.ndarray:
    """Extract a normalized feature vector from a square image.

    Subtracts mean and divides by std to remove background color influence.
    """
    inner = _inner_crop(square, 0.65)
    if inner.size == 0:
        return np.zeros(CANONICAL * CANONICAL)

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
    resized = cv2.resize(gray, (CANONICAL, CANONICAL)).astype(float)

    mean = np.mean(resized)
    std = np.std(resized)
    if std < 1e-6:
        return np.zeros(CANONICAL * CANONICAL)

    normalized = (resized - mean) / std
    return normalized.flatten()


def read_position(frame: np.ndarray) -> str | None:
    """Read chess position using intra-frame clustering."""
    grid = find_board_in_frame(frame)
    if grid is None:
        return None
    v_lines, h_lines, sq_size = grid
    squares = crop_squares(frame, v_lines, h_lines)
    light_bgr, dark_bgr = detect_theme(squares)

    # Extract features for all 64 squares
    features = []
    positions = []
    for r in range(8):
        for c in range(8):
            feat = _square_feature(squares[r][c])
            features.append(feat)
            positions.append((r, c))

    features = np.array(features)  # (64, CANONICAL^2)

    # Compute pairwise distances
    dists = pdist(features, metric="euclidean")

    # Hierarchical clustering
    Z = linkage(dists, method="complete")

    # Try different numbers of clusters and pick the one that best
    # separates empty from occupied
    best_labels = None
    best_score = -1

    for n_clusters in range(3, 16):
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        # Score: how well do the clusters separate by variance?
        # Empty squares should form tight clusters (low intra-cluster variance)
        cluster_vars: dict[int, list[float]] = {}
        for i, (r, c) in enumerate(positions):
            inner = _inner_crop(squares[r][c], 0.6)
            if inner.size == 0:
                continue
            gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
            var = float(np.var(gray))
            cluster_vars.setdefault(labels[i], []).append(var)

        # The best clustering has clearly separated empty clusters (low mean var)
        # from occupied clusters (high mean var)
        cluster_mean_vars = {k: np.mean(v) for k, v in cluster_vars.items()}
        all_vars = list(cluster_mean_vars.values())
        if len(all_vars) < 2:
            continue

        all_vars_sorted = sorted(all_vars)
        # Find the biggest gap
        max_gap = 0
        for i in range(len(all_vars_sorted) - 1):
            gap = all_vars_sorted[i + 1] - all_vars_sorted[i]
            if gap > max_gap:
                max_gap = gap

        score = max_gap * n_clusters  # Prefer clear separation, moderate cluster count
        if score > best_score:
            best_score = score
            best_labels = labels.copy()
            best_cluster_vars = cluster_mean_vars.copy()

    if best_labels is None:
        return None

    # Identify empty clusters: those with lowest mean variance
    var_threshold = np.median(list(best_cluster_vars.values()))
    empty_clusters = {k for k, v in best_cluster_vars.items() if v < var_threshold}

    # Classify squares
    piece_grid: list[list[str | None]] = [[None] * 8 for _ in range(8)]
    occupied_info: list[tuple[int, int, int, float]] = []  # (r, c, cluster, brightness)

    for i, (r, c) in enumerate(positions):
        cluster = best_labels[i]
        if cluster in empty_clusters:
            continue  # empty

        inner = _inner_crop(squares[r][c], 0.7)
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if len(inner.shape) == 3 else inner
        brightness = float(np.mean(gray))
        occupied_info.append((r, c, cluster, brightness))

    if not occupied_info:
        return None

    # Classify piece color by brightness relative to theme
    light_bright = float(
        np.mean(cv2.cvtColor(light_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY))
    )
    dark_bright = float(
        np.mean(cv2.cvtColor(dark_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY))
    )
    theme_mid = (light_bright + dark_bright) / 2.0

    # Group clusters by piece color
    cluster_brightnesses: dict[int, list[float]] = {}
    for r, c, cluster, brightness in occupied_info:
        cluster_brightnesses.setdefault(cluster, []).append(brightness)

    white_clusters = set()
    black_clusters = set()
    for cluster, brights in cluster_brightnesses.items():
        if np.mean(brights) > theme_mid:
            white_clusters.add(cluster)
        else:
            black_clusters.add(cluster)

    # Within each color, assign piece types by cluster size
    for color_clusters, is_white in [(white_clusters, True), (black_clusters, False)]:
        # Sort clusters by size (largest = most common = pawns)
        cluster_members: dict[int, list[tuple[int, int]]] = {}
        for r, c, cluster, brightness in occupied_info:
            if cluster in color_clusters:
                cluster_members.setdefault(cluster, []).append((r, c))

        if not cluster_members:
            continue

        # Sort by cluster size descending
        sorted_clusters = sorted(cluster_members.items(), key=lambda x: -len(x[1]))

        # Assign piece types:
        # Largest cluster → pawns
        # Remaining clusters → non-pawn pieces
        piece_types = ["p", "r", "n", "b", "q", "k"]  # priority order by cluster size
        type_idx = 0

        for cluster_id, members in sorted_clusters:
            if type_idx >= len(piece_types):
                pt = "p"  # overflow → pawn
            else:
                pt = piece_types[type_idx]
            type_idx += 1

            for r, c in members:
                sym = pt.upper() if is_white else pt
                # Pawns can't be on rows 0 or 7
                if pt == "p" and r in (0, 7):
                    sym = "N" if is_white else "n"
                piece_grid[r][c] = sym

    # Ensure one king per side
    for king_sym, color_name in [("K", "white"), ("k", "black")]:
        has_king = any(piece_grid[r][c] == king_sym for r in range(8) for c in range(8))
        if not has_king:
            # Find a non-pawn piece of this color and make it king
            target_clusters = white_clusters if color_name == "white" else black_clusters
            # Pick the piece in the smallest cluster (most unique = likely king)
            candidates = []
            for r, c, cluster, brightness in occupied_info:
                if cluster in target_clusters and piece_grid[r][c] is not None:
                    p = piece_grid[r][c]
                    is_right_color = p.isupper() if color_name == "white" else p.islower()
                    if is_right_color and p.lower() != "p":
                        candidates.append((r, c))
            if candidates:
                r, c = candidates[0]
                piece_grid[r][c] = king_sym
            elif occupied_info:
                # Last resort: any piece of this color
                for r, c, cluster, brightness in occupied_info:
                    if cluster in target_clusters and piece_grid[r][c] is not None:
                        piece_grid[r][c] = king_sym
                        break

    flipped = detect_orientation(piece_grid)
    board = build_board(piece_grid, flipped=flipped)
    return board.board_fen()
