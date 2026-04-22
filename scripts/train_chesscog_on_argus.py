"""Train chesscog-style occupancy + piece classifiers on argus data.

Pipeline:
  1. Export argus annotations to chesscog's on-disk ImageFolder layout
     (skipped if target dir already has the expected class folders).
  2. For each task (occupancy, piece): load chesscog's ResNet.yaml config,
     override DATASET.PATH to the exported root, patch chesscog.core.DEVICE
     to use MPS/CUDA, and call chesscog.core.training.train.train(cfg, run_dir)
     directly.

Bypasses chesscog's CLI entrypoint so we don't have to deal with recap URI
registrations for data:// / models:// / runs:// paths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from pipeline.physical.chesscog_baseline import (
    DEFAULT_CHESSCOG_ROOT,
    ensure_chesscog_on_path,
)
from pipeline.physical.chesscog_baseline.dataset_export import export_split

ensure_chesscog_on_path()


def _resolve_device(preferred: str | None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _patch_chesscog_device(device: torch.device) -> None:
    """Point chesscog.core.DEVICE + device() default at our target device.

    chesscog.core.device has `dev: str = DEVICE` which binds the default at
    function definition. Patching the module constant alone isn't enough —
    also update the function's __defaults__ tuple.
    """
    import chesscog.core

    dev_str = str(device)
    chesscog.core.DEVICE = dev_str
    chesscog.core.device.__defaults__ = (dev_str,)


def _needs_export(exported_root: Path) -> bool:
    """Quick check: if both class folders exist under occupancy/{train,val} and
    pieces/{train,val}, assume the export is already done."""
    required = [
        exported_root / "occupancy" / split / cls
        for split in ("train", "val")
        for cls in ("empty", "occupied")
    ] + [exported_root / "pieces" / split / "white_pawn" for split in ("train", "val")]
    return not all(p.exists() for p in required)


def _run_export(data_root: Path, exported_root: Path) -> None:
    for split in ("train", "val"):
        export_split(
            annotation_root=data_root / split,
            output_root=exported_root,
            split_name=split,
        )


def _train_one(
    task: str,
    cfg_path: Path,
    dataset_path: Path,
    run_dir: Path,
    device: torch.device,
) -> Path:
    """Train chesscog's model for one task. Returns the trained checkpoint path."""
    from recap import CfgNode

    # chesscog.core.training.train references the module-level DEVICE constant
    # via `from chesscog.core import device` (the helper). Re-patching here is
    # defensive in case something else already imported the submodule.
    _patch_chesscog_device(device)

    from chesscog.core.training.train import train

    print(f"[{task}] loading {cfg_path}")
    cfg = CfgNode.load_yaml_with_base(str(cfg_path))
    cfg.DATASET.PATH = str(dataset_path)
    # Shrink workers — defaults to 2 which is fine, but keep explicit.
    print(f"[{task}] dataset path: {cfg.DATASET.PATH}")
    print(f"[{task}] classes: {list(cfg.DATASET.CLASSES)}")
    print(f"[{task}] phases: {[dict(p) for p in cfg.TRAINING.PHASES]}")
    print(f"[{task}] run_dir: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    train(cfg, run_dir)
    ckpt = run_dir / f"{run_dir.name}.pt"
    assert ckpt.exists(), f"expected checkpoint at {ckpt}"
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/physical"))
    ap.add_argument(
        "--exported-root",
        type=Path,
        default=Path("data/chesscog_baseline"),
    )
    ap.add_argument(
        "--weights-root",
        type=Path,
        default=Path("weights/chesscog_baseline"),
    )
    ap.add_argument("--chesscog-root", type=Path, default=DEFAULT_CHESSCOG_ROOT)
    ap.add_argument("--device", type=str, default=None, help="mps|cuda|cpu")
    ap.add_argument(
        "--stage",
        choices=("export", "train", "all"),
        default="all",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["occupancy", "piece"],
        choices=("occupancy", "piece"),
    )
    ap.add_argument(
        "--force-export", action="store_true", help="Re-export even if target dirs exist"
    )
    args = ap.parse_args()

    device = _resolve_device(args.device)
    print(f"using device: {device}")

    if args.stage in ("export", "all"):
        if args.force_export or _needs_export(args.exported_root):
            _run_export(args.data_root, args.exported_root)
        else:
            print(f"skipping export (target root {args.exported_root} looks complete)")

    if args.stage in ("train", "all"):
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        task_to_cfg = {
            "occupancy": args.chesscog_root / "config/occupancy_classifier/ResNet.yaml",
            "piece": args.chesscog_root / "config/piece_classifier/ResNet.yaml",
        }
        task_to_subdir = {"occupancy": "occupancy", "piece": "pieces"}

        for task in args.tasks:
            run_name = f"ResNet_{timestamp}"
            run_dir = args.weights_root / task / run_name
            ckpt = _train_one(
                task=task,
                cfg_path=task_to_cfg[task],
                dataset_path=(args.exported_root / task_to_subdir[task]).resolve(),
                run_dir=run_dir.resolve(),
                device=device,
            )
            print(f"[{task}] trained checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
