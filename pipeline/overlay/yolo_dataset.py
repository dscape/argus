"""Export overlay bbox training annotations as a YOLO detection dataset."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from pipeline.paths import GROUND_TRUTH_PATH, PROJECT_ROOT, VIDEOS_DIR, find_frame

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "frames"
FIXTURE_GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "overlay" / "yolo"
DATASET_CLASS_NAME = "overlay"
_LEGACY_GT_PATH = PROJECT_ROOT / "data" / "overlay" / "dataset" / "frames" / "ground_truth.json"


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
    source_ground_truth = source_ground_truth.resolve()
    fixture_ground_truth = fixture_ground_truth.resolve()
    legacy_fallback = _LEGACY_GT_PATH if source_ground_truth == GROUND_TRUTH_PATH.resolve() else None
    all_gt = _load_ground_truth(source_ground_truth, legacy_fallback=legacy_fallback)
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

    fixture_video_ids = {_video_id(key) for key in fixture_gt}
    train_val_keys = [key for key in all_gt if _video_id(key) not in fixture_video_ids]
    train_keys, val_keys = _split_train_val(train_val_keys, all_gt, val_fraction, seed)
    test_keys = sorted(fixture_gt)

    manifest = {
        "source_ground_truth": str(source_ground_truth),
        "fixture_ground_truth": str(fixture_ground_truth),
        "class_name": DATASET_CLASS_NAME,
        "splits": {
            "train": _export_split(
                train_keys,
                all_gt,
                VIDEOS_DIR,
                split_dirs["train"],
                resolve_from_video_store=True,
            ),
            "val": _export_split(
                val_keys,
                all_gt,
                VIDEOS_DIR,
                split_dirs["val"],
                resolve_from_video_store=True,
            ),
            "test": _export_split(test_keys, fixture_gt, FIXTURES_DIR, split_dirs["test"]),
        },
    }

    _validate_train_split(manifest["splits"]["train"])

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


def _load_ground_truth(
    path: Path,
    *,
    legacy_fallback: Path | None = None,
) -> dict[str, dict[str, object]]:
    load_path = path
    if not load_path.exists():
        if legacy_fallback is not None and legacy_fallback.exists():
            load_path = legacy_fallback
        else:
            raise FileNotFoundError(f"Ground truth file not found: {path}")

    data = json.loads(load_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping ground truth file at {load_path}")
    return data


def _split_train_val(
    keys: list[str],
    ground_truth: dict[str, dict[str, object]],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    keys_by_video: dict[str, list[str]] = {}
    for key in keys:
        keys_by_video.setdefault(_video_id(key), []).append(key)

    positive_videos = [
        video_id
        for video_id, video_keys in keys_by_video.items()
        if any(bool(ground_truth[key].get("has_overlay")) for key in video_keys)
    ]
    negative_videos = [
        video_id
        for video_id, video_keys in keys_by_video.items()
        if not any(bool(ground_truth[key].get("has_overlay")) for key in video_keys)
    ]
    rng.shuffle(positive_videos)
    rng.shuffle(negative_videos)

    val_pos = _split_count(len(positive_videos), val_fraction)
    val_neg = _split_count(len(negative_videos), val_fraction)
    val_videos = set(positive_videos[:val_pos] + negative_videos[:val_neg])

    train_keys = sorted(
        key
        for video_id, video_keys in keys_by_video.items()
        if video_id not in val_videos
        for key in video_keys
    )
    val_keys = sorted(
        key
        for video_id, video_keys in keys_by_video.items()
        if video_id in val_videos
        for key in video_keys
    )
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
    *,
    resolve_from_video_store: bool = False,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    for key in keys:
        video_id, label = key.split("/", 1)
        annotation = ground_truth[key]
        source_image = _resolve_annotation_image(
            images_root,
            video_id,
            label,
            annotation,
            resolve_from_video_store=resolve_from_video_store,
        )
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
    *,
    resolve_from_video_store: bool = False,
) -> Path:
    image_rel = annotation.get("image")
    if isinstance(image_rel, str):
        direct = images_root / image_rel
        if direct.exists():
            return direct

    expected_size = _annotation_size(annotation)
    candidates: list[Path] = [images_root / video_id / f"{label}.jpg"]
    if resolve_from_video_store:
        for tier in ("hires", "fullres", "lores"):
            candidate = find_frame(video_id, tier, label)
            if candidate is not None:
                candidates.append(candidate)

    existing = _unique_existing_paths(candidates)
    if expected_size is not None:
        for candidate in existing:
            if _image_size(candidate) == expected_size:
                return candidate

    if existing:
        return existing[0]

    raise FileNotFoundError(f"Could not resolve image for {video_id}/{label}")


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        result.append(path)
    return result


def _video_id(key: str) -> str:
    return key.split("/", 1)[0]


def _validate_train_split(entries: list[dict[str, object]]) -> None:
    if not entries:
        raise ValueError("YOLO export produced an empty train split")
    if not any(bool(entry["has_overlay"]) for entry in entries):
        raise ValueError("YOLO export train split has no positive overlay labels")


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
