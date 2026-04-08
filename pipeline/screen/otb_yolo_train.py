"""Train the default YOLO OTB-board detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.paths import PROJECT_ROOT

DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs" / "otb_yolo"


@dataclass(frozen=True)
class YoloTrainResult:
    best_weights: Path
    save_dir: Path


def train_otb_yolo(
    dataset_yaml: Path,
    *,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project: Path = DEFAULT_RUNS_DIR,
    name: str = "train",
) -> YoloTrainResult:
    """Train YOLO on the exported OTB-board dataset."""
    from ultralytics import YOLO

    dataset_yaml = dataset_yaml.resolve()
    project = project.resolve()

    resolved_device = _resolve_device(device)
    model = YOLO(model_name)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        project=str(project),
        name=name,
        exist_ok=True,
        workers=0,
        verbose=True,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics training finished without exposing trainer state")

    best = Path(str(trainer.best)).resolve()
    save_dir = Path(str(trainer.save_dir)).resolve()
    if not best.exists():
        raise FileNotFoundError(f"Expected best weights at {best}")

    return YoloTrainResult(best_weights=best, save_dir=save_dir)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
