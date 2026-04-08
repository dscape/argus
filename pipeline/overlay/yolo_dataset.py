"""Export overlay bbox training annotations as a YOLO detection dataset."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from pipeline.paths import GROUND_TRUTH_PATH, PROJECT_ROOT, VIDEOS_DIR

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "frames"
FIXTURE_GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "overlay" / "yolo"
DATASET_CLASS_NAME = "overlay"


@dataclass(frozen=True)
class SplitCounts:
    images: int
    positives: int
    negatives: int


@dataclass(frozen=True)
class YoloDatasetExport:
    dataset_dir: Path
    dataset_yaml: Path
    manifest_path: Path
    train: SplitCounts
    val: SplitCounts
    test: SplitCounts


def export_overlay_yolo_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    source_ground_truth: Path = GROUND_TRUTH_PATH,
    fixture_ground_truth: Path = FIXTURE_GROUND_TRUTH_PATH,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> YoloDatasetExport:
    """Export overlay training annotations into YOLO train/val/test directories.

    Training data comes from ``data/videos/ground_truth.json`` excluding any
    keys already committed as test fixtures. The committed fixture set becomes
    the YOLO ``test`` split so detector training is evaluated against the same
    frames used by the runtime overlay detector tests.
    """
    all_gt = _load_ground_truth(source_ground_truth)
    fixture_gt = _load_ground_truth(fixture_ground_truth)

    dataset_dir = dataset_dir.resolve()
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    split_dirs = {
        split: {
            "images": dataset_dir / "images" / split,
            "labels": dataset_dir / "labels" / split,
        }
        for split in ("train", "val", "test")
    }
    for dirs in split_dirs.values():
        dirs["images"].mkdir(parents=True, exist_ok=True)
        dirs["labels"].mkdir(parents=True, exist_ok=True)

    train_val_keys = [key for key in all_gt if key not in fixture_gt]
    train_keys, val_keys = _split_train_val(train_val_keys, all_gt, val_fraction, seed)
    test_keys = sorted(fixture_gt)

    manifest = {
        "source_ground_truth": str(source_ground_truth),
        "fixture_ground_truth": str(fixture_ground_truth),
        "class_name": DATASET_CLASS_NAME,
        "splits": {
            "train": _export_split(train_keys, all_gt, VIDEOS_DIR, split_dirs["train"]),
            "val": _export_split(val_keys, all_gt, VIDEOS_DIR, split_dirs["val"]),
            "test": _export_split(test_keys, fixture_gt, FIXTURES_DIR, split_dirs["test"]),
        },
    }

    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(_build_dataset_yaml(dataset_dir))

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return YoloDatasetExport(
        dataset_dir=dataset_dir,
        dataset_yaml=dataset_yaml,
        manifest_path=manifest_path,
        train=_counts_from_manifest(manifest["splits"]["train"]),
        val=_counts_from_manifest(manifest["splits"]["val"]),
        test=_counts_from_manifest(manifest["splits"]["test"]),
    )


def _counts_from_manifest(entries: list[dict[str, object]]) -> SplitCounts:
    positives = sum(1 for entry in entries if bool(entry["has_overlay"]))
    negatives = len(entries) - positives
    return SplitCounts(images=len(entries), positives=positives, negatives=negatives)


def _load_ground_truth(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping ground truth file at {path}")
    return data


def _split_train_val(
    keys: list[str],
    ground_truth: dict[str, dict[str, object]],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    positives = [key for key in keys if bool(ground_truth[key].get("has_overlay"))]
    negatives = [key for key in keys if not bool(ground_truth[key].get("has_overlay"))]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    val_pos = _split_count(len(positives), val_fraction)
    val_neg = _split_count(len(negatives), val_fraction)

    val_keys = sorted(positives[:val_pos] + negatives[:val_neg])
    train_keys = sorted(positives[val_pos:] + negatives[val_neg:])
    return train_keys, val_keys


def _split_count(size: int, fraction: float) -> int:
    if size == 0 or fraction <= 0:
        return 0
    if size == 1:
        return 1
    return max(1, min(size - 1, int(round(size * fraction))))


def _export_split(
    keys: list[str],
    ground_truth: dict[str, dict[str, object]],
    images_root: Path,
    split_dirs: dict[str, Path],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    for key in keys:
        video_id, label = key.split("/", 1)
        annotation = ground_truth[key]
        source_image = _resolve_annotation_image(images_root, video_id, label, annotation)
        image_name = f"{video_id}__{label}.jpg"
        image_dest = split_dirs["images"] / image_name
        label_dest = split_dirs["labels"] / f"{video_id}__{label}.txt"

        shutil.copy2(source_image, image_dest)
        if bool(annotation.get("has_overlay")):
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Positive annotation missing bbox for {key}")
            label_dest.write_text(_to_yolo_label(annotation, bbox))
        elif label_dest.exists():
            label_dest.unlink()

        entries.append(
            {
                "key": key,
                "image": str(image_dest.relative_to(split_dirs["images"].parent.parent)),
                "label": (
                    str(label_dest.relative_to(split_dirs["labels"].parent.parent))
                    if bool(annotation.get("has_overlay"))
                    else None
                ),
                "has_overlay": bool(annotation.get("has_overlay")),
                "source_image": str(source_image),
            }
        )

    return entries


def _resolve_annotation_image(
    images_root: Path,
    video_id: str,
    label: str,
    annotation: dict[str, object],
) -> Path:
    image_rel = annotation.get("image")
    if isinstance(image_rel, str):
        direct = images_root / image_rel
        if direct.exists():
            return direct

    expected_size = _annotation_size(annotation)
    candidates = [
        VIDEOS_DIR / video_id / "hires" / f"{label}.jpg",
        VIDEOS_DIR / video_id / "fullres" / f"{label}.jpg",
        VIDEOS_DIR / video_id / "lores" / f"{label}.jpg",
        images_root / video_id / f"{label}.jpg",
    ]

    existing = [path for path in candidates if path.exists()]
    if expected_size is not None:
        for candidate in existing:
            if _image_size(candidate) == expected_size:
                return candidate

    if existing:
        return existing[0]

    raise FileNotFoundError(f"Could not resolve image for {video_id}/{label}")


def _annotation_size(annotation: dict[str, object]) -> tuple[int, int] | None:
    width = annotation.get("frame_width")
    height = annotation.get("frame_height")
    if isinstance(width, int) and isinstance(height, int):
        return width, height
    return None


def _image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    return w, h


def _to_yolo_label(annotation: dict[str, object], bbox: list[object]) -> str:
    frame_width = annotation.get("frame_width")
    frame_height = annotation.get("frame_height")
    if not isinstance(frame_width, int) or not isinstance(frame_height, int):
        raise ValueError("YOLO export requires frame_width/frame_height")

    x, y, w, h = bbox
    if not all(isinstance(value, int) for value in (x, y, w, h)):
        raise ValueError(f"Expected integer bbox, got {bbox}")

    x_center = (x + w / 2) / frame_width
    y_center = (y + h / 2) / frame_height
    width = w / frame_width
    height = h / frame_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def _build_dataset_yaml(dataset_dir: Path) -> str:
    root = dataset_dir.resolve()
    return "\n".join(
        [
            f"path: {root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            f"  0: {DATASET_CLASS_NAME}",
            "",
        ]
    )
