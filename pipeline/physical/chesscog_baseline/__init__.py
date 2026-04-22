"""Chesscog-baseline classifiers trained on argus data.

Mirrors chesscog's two-stage recipe (top-down homography warp + ResNet18 +
fixed per-rank/file crop extensions for the piece classifier) but trains on
argus's real photos instead of chesscog's synthetic renders. Exists so we can
A/B the chesscog approach against argus's current DINOv2 + 3D-box-projection
classifiers on the same validation set.

Requires chesscog on sys.path (via `ensure_chesscog_on_path()`) and the
RECAP_CONFIG env var pointing at `~/dev/chesscog/config/` so chesscog's YAML
`_BASE_` inheritance resolves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_CHESSCOG_ROOT = Path.home() / "dev/chesscog"


def ensure_chesscog_on_path(chesscog_root: Path | str = DEFAULT_CHESSCOG_ROOT) -> Path:
    """Prepend chesscog's repo root to sys.path and set RECAP_CONFIG.

    Idempotent. Returns the resolved chesscog root.
    """
    root = Path(chesscog_root).expanduser().resolve()
    if not (root / "chesscog" / "__init__.py").exists():
        raise FileNotFoundError(f"chesscog not found at {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ.setdefault("RECAP_CONFIG", str(root / "config"))
    return root


# Argus class id (1..12) → chesscog piece folder name.
# Argus SQUARE_CLASS_NAMES[1..12] = [P, N, B, R, Q, K, p, n, b, r, q, k].
ARGUS_CLASS_TO_CHESSCOG_FOLDER: dict[int, str] = {
    1: "white_pawn",
    2: "white_knight",
    3: "white_bishop",
    4: "white_rook",
    5: "white_queen",
    6: "white_king",
    7: "black_pawn",
    8: "black_knight",
    9: "black_bishop",
    10: "black_rook",
    11: "black_queen",
    12: "black_king",
}

# Inverse: chesscog piece folder name → argus class id.
CHESSCOG_FOLDER_TO_ARGUS_CLASS: dict[str, int] = {
    v: k for k, v in ARGUS_CLASS_TO_CHESSCOG_FOLDER.items()
}


def build_chesscog_to_argus_remap(chesscog_classes: list[str]) -> list[int]:
    """Build a lookup `chesscog_idx → argus_class_id` for the piece classifier.

    chesscog_classes is cfg.DATASET.CLASSES from the piece classifier config,
    ordered alphabetically by chesscog (e.g. [black_bishop, black_king, ...]).
    Return a list where index = chesscog idx, value = argus class id in [1, 12].
    """
    remap = [CHESSCOG_FOLDER_TO_ARGUS_CLASS[name] for name in chesscog_classes]
    assert len(remap) == 12, f"expected 12 piece classes, got {len(remap)}"
    return remap
