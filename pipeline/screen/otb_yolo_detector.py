"""Default YOLO-based OTB-board detector used by calibration and screening."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pipeline.paths import PROJECT_ROOT
from pipeline.runtime_assets import OTB_YOLO_WEIGHTS, RuntimeAsset, ensure_runtime_asset

if TYPE_CHECKING:
    from pipeline.screen.dual_region_detector import OTBDetection

MODEL_CODE_VERSION = "v1"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "otb_yolo"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
_RETRY_IMGSZS = (1280,)


@lru_cache(maxsize=4)
def _load_model(weights_path: str):
    from ultralytics import YOLO

    return YOLO(weights_path)


@lru_cache(maxsize=8)
def _validated_weights_path(weights_path: str) -> Path:
    candidate = Path(weights_path).expanduser().resolve()
    if candidate == DEFAULT_WEIGHTS_PATH.resolve():
        return ensure_runtime_asset(OTB_YOLO_WEIGHTS)

    return ensure_runtime_asset(
        RuntimeAsset(name="OTB YOLO detector weights", path=candidate)
    )


def detect_otb_yolo(
    frame: np.ndarray,
    *,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    conf: float = 0.20,
    imgsz: int = 640,
    device: str = "auto",
) -> OTBDetection:
    """Run YOLO detection and return the best OTB-board bbox, if any.

    The default 640px input can miss small, perspective OTB boards in wide
    1080p shots. Retry once at a higher resolution on misses so calibration can
    recover those clips without paying the larger cost on every frame.
    """
    h, w = frame.shape[:2]
    resolution = (w, h)

    model = _load_model(str(_resolve_weights_path(weights_path)))
    resolved_device = _resolve_device(device)

    for candidate_imgsz in _candidate_imgszs(imgsz):
        detection = _predict_otb_bbox(
            model,
            frame,
            conf=conf,
            imgsz=candidate_imgsz,
            device=resolved_device,
            resolution=resolution,
        )
        if detection.found:
            return detection

    from pipeline.screen.dual_region_detector import OTBDetection

    return OTBDetection(found=False, frame_resolution=resolution)


def _candidate_imgszs(imgsz: int) -> tuple[int, ...]:
    candidates = [imgsz]
    for retry_imgsz in _RETRY_IMGSZS:
        if retry_imgsz > imgsz:
            candidates.append(retry_imgsz)
    return tuple(candidates)


def _predict_otb_bbox(
    model,
    frame: np.ndarray,
    *,
    conf: float,
    imgsz: int,
    device: str,
    resolution: tuple[int, int],
):
    from pipeline.screen.dual_region_detector import OTBDetection

    h, w = frame.shape[:2]
    result = model.predict(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        max_det=1,
        verbose=False,
    )[0]

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return OTBDetection(found=False, frame_resolution=resolution)

    best_idx = int(boxes.conf.argmax().item())
    xyxy = boxes.xyxy[best_idx].tolist()
    score = float(boxes.conf[best_idx].item())

    x1, y1, x2, y2 = xyxy
    x = max(0, min(int(round(x1)), w - 1))
    y = max(0, min(int(round(y1)), h - 1))
    x2_int = max(x + 1, min(int(round(x2)), w))
    y2_int = max(y + 1, min(int(round(y2)), h))
    bbox = (x, y, x2_int - x, y2_int - y)

    return OTBDetection(
        found=True,
        confidence=score,
        bbox=bbox,
        frame_resolution=resolution,
    )


def _resolve_weights_path(weights_path: str | Path) -> Path:
    return _validated_weights_path(str(weights_path))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
