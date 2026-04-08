"""Export an OTB-board YOLO dataset from cached screening frames."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pipeline.db.connection import get_conn
from pipeline.paths import PROJECT_ROOT, find_frame

DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "screening" / "otb_yolo"
DATASET_CLASS_NAME = "otb_board"
FRAME_LABELS = ("25pct", "50pct", "75pct")
PREFERRED_TIERS = ("hires", "lores")


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


@dataclass(frozen=True)
class FrameCandidate:
    video_id: str
    label: str
    image_path: Path
    tier: str


@dataclass(frozen=True)
class PseudoLabel:
    bbox: tuple[int, int, int, int]
    confidence: float
    source: str


class PseudoLabeler(Protocol):
    def label_image(self, image: np.ndarray) -> PseudoLabel | None:
        ...


class OwlV2BoardLabeler:
    """Pseudo-label OTB boards with OWLv2."""

    def __init__(
        self,
        *,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        device: str = "auto",
        threshold: float = 0.05,
        nms_iou: float = 0.35,
        min_side_fraction: float = 0.12,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.min_side_fraction = min_side_fraction
        self.prompts = [["chessboard", "physical chessboard", "wooden chessboard"]]
        self._processor = None
        self._model = None

    def label_image(self, image: np.ndarray) -> PseudoLabel | None:
        import torch
        from PIL import Image
        from torchvision.ops import nms
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if self._processor is None or self._model is None:
            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()

        h, w = image.shape[:2]
        min_side = max(1.0, min(h, w) * self.min_side_fraction)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        inputs = self._processor(text=self.prompts, images=pil_image, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([(h, w)], device=self.device),
            threshold=self.threshold,
            text_labels=self.prompts,
        )[0]
        boxes = results["boxes"].detach().cpu()
        scores = results["scores"].detach().cpu()
        if len(boxes) == 0:
            return None

        keep = nms(boxes, scores, self.nms_iou)
        best_rank = -1.0
        best_bbox: tuple[int, int, int, int] | None = None
        best_score = 0.0

        for idx in keep.tolist():
            x1, y1, x2, y2 = boxes[idx].tolist()
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            if bw < min_side or bh < min_side:
                continue
            rank = float(scores[idx]) * (bw * bh) ** 0.5
            if rank <= best_rank:
                continue

            x = max(0, min(int(round(x1)), w - 1))
            y = max(0, min(int(round(y1)), h - 1))
            x2_int = max(x + 1, min(int(round(x2)), w))
            y2_int = max(y + 1, min(int(round(y2)), h))
            best_rank = rank
            best_score = float(scores[idx])
            best_bbox = (x, y, x2_int - x, y2_int - y)

        if best_bbox is None:
            return None

        return PseudoLabel(
            bbox=best_bbox,
            confidence=best_score,
            source=f"owlv2:{self.model_name}",
        )


def export_otb_yolo_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    positives: list[FrameCandidate] | None = None,
    negatives: list[FrameCandidate] | None = None,
    labeler: PseudoLabeler | None = None,
    positive_video_limit: int | None = None,
    negative_video_limit: int | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> YoloDatasetExport:
    """Export a YOLO dataset for OTB-board detection."""
    if positives is None or negatives is None:
        positives, negatives = collect_otb_yolo_candidates(
            positive_video_limit=positive_video_limit,
            negative_video_limit=negative_video_limit,
            seed=seed,
        )

    labeler = labeler or OwlV2BoardLabeler()
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

    positive_splits = _split_candidates_by_video(positives, val_fraction, test_fraction, seed)
    negative_splits = _split_candidates_by_video(negatives, val_fraction, test_fraction, seed + 1)

    manifest = {
        "class_name": DATASET_CLASS_NAME,
        "label_source": type(labeler).__name__,
        "splits": {
            split: _export_split(
                positive_splits[split],
                negative_splits[split],
                split_dirs[split],
                labeler,
            )
            for split in ("train", "val", "test")
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


def collect_otb_yolo_candidates(
    *,
    positive_video_limit: int | None = None,
    negative_video_limit: int | None = None,
    seed: int = 42,
) -> tuple[list[FrameCandidate], list[FrameCandidate]]:
    """Collect cached positive and negative frame candidates from the DB."""
    positive_video_ids = _sample_video_ids(
        _query_video_ids(
            """
            SELECT video_id
            FROM youtube_videos
            WHERE screening_status = 'approved'
              AND layout_type = 'otb_only'
            ORDER BY published_at DESC NULLS LAST
            """
        ),
        positive_video_limit,
        seed,
    )

    if negative_video_limit is None and positive_video_limit is not None:
        negative_video_limit = positive_video_limit
    if negative_video_limit is None:
        negative_video_limit = len(positive_video_ids)

    negative_video_ids = _sample_video_ids(
        _query_video_ids(
            """
            SELECT video_id
            FROM youtube_videos
            WHERE screening_status = 'rejected'
            ORDER BY published_at DESC NULLS LAST
            """
        ),
        negative_video_limit,
        seed + 1,
    )

    positives = _collect_frame_candidates(positive_video_ids)
    negatives = _collect_frame_candidates(negative_video_ids)
    return positives, negatives


def _query_video_ids(sql: str) -> list[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]


def _sample_video_ids(video_ids: list[str], limit: int | None, seed: int) -> list[str]:
    if limit is None or limit >= len(video_ids):
        return video_ids
    rng = random.Random(seed)
    sampled = list(video_ids)
    rng.shuffle(sampled)
    return sampled[:limit]


def _collect_frame_candidates(video_ids: list[str]) -> list[FrameCandidate]:
    candidates: list[FrameCandidate] = []
    for video_id in video_ids:
        for label in FRAME_LABELS:
            resolved = _resolve_candidate_frame(video_id, label)
            if resolved is None:
                continue
            image_path, tier = resolved
            candidates.append(
                FrameCandidate(
                    video_id=video_id,
                    label=label,
                    image_path=image_path,
                    tier=tier,
                )
            )
    return candidates


def _resolve_candidate_frame(video_id: str, label: str) -> tuple[Path, str] | None:
    for tier in PREFERRED_TIERS:
        candidate = find_frame(video_id, tier, label)
        if candidate is not None:
            return candidate, tier
    return None


def _split_candidates_by_video(
    candidates: list[FrameCandidate],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[FrameCandidate]]:
    video_ids = sorted({candidate.video_id for candidate in candidates})
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    test_count = _split_count(len(video_ids), test_fraction)
    remaining = video_ids[test_count:]
    val_count = _split_count(len(remaining), val_fraction)

    split_by_video = {
        "test": set(video_ids[:test_count]),
        "val": set(remaining[:val_count]),
    }
    split_by_video["train"] = set(remaining[val_count:])

    return {
        split: [candidate for candidate in candidates if candidate.video_id in split_videos]
        for split, split_videos in split_by_video.items()
    }


def _split_count(size: int, fraction: float) -> int:
    if size == 0 or fraction <= 0:
        return 0
    if size == 1:
        return 1
    return max(1, min(size - 1, int(round(size * fraction))))


def _export_split(
    positives: list[FrameCandidate],
    negatives: list[FrameCandidate],
    split_dirs: dict[str, Path],
    labeler: PseudoLabeler,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    for candidate in positives:
        image = cv2.imread(str(candidate.image_path))
        if image is None:
            continue

        pseudo_label = labeler.label_image(image)
        if pseudo_label is None:
            continue

        image_dest, label_dest = _destination_paths(candidate, split_dirs)
        shutil.copy2(candidate.image_path, image_dest)
        label_dest.write_text(_to_yolo_label(image.shape, pseudo_label.bbox))
        entries.append(
            {
                "video_id": candidate.video_id,
                "label": candidate.label,
                "image": str(image_dest.relative_to(split_dirs["images"].parent.parent)),
                "label_path": str(label_dest.relative_to(split_dirs["labels"].parent.parent)),
                "has_otb_board": True,
                "bbox": list(pseudo_label.bbox),
                "confidence": round(pseudo_label.confidence, 4),
                "source_image": str(candidate.image_path),
                "source_tier": candidate.tier,
                "label_source": pseudo_label.source,
            }
        )

    for candidate in negatives:
        image_dest, _ = _destination_paths(candidate, split_dirs)
        shutil.copy2(candidate.image_path, image_dest)
        entries.append(
            {
                "video_id": candidate.video_id,
                "label": candidate.label,
                "image": str(image_dest.relative_to(split_dirs["images"].parent.parent)),
                "label_path": None,
                "has_otb_board": False,
                "bbox": None,
                "confidence": None,
                "source_image": str(candidate.image_path),
                "source_tier": candidate.tier,
                "label_source": None,
            }
        )

    return entries


def _destination_paths(candidate: FrameCandidate, split_dirs: dict[str, Path]) -> tuple[Path, Path]:
    image_name = f"{candidate.video_id}__{candidate.label}.jpg"
    return (
        split_dirs["images"] / image_name,
        split_dirs["labels"] / f"{candidate.video_id}__{candidate.label}.txt",
    )


def _to_yolo_label(
    image_shape: tuple[int, ...],
    bbox: tuple[int, int, int, int],
) -> str:
    frame_height, frame_width = image_shape[:2]
    x, y, w, h = bbox
    x_center = (x + w / 2) / frame_width
    y_center = (y + h / 2) / frame_height
    width = w / frame_width
    height = h / frame_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def _counts_from_manifest(entries: list[dict[str, object]]) -> SplitCounts:
    positives = sum(1 for entry in entries if bool(entry["has_otb_board"]))
    negatives = len(entries) - positives
    return SplitCounts(images=len(entries), positives=positives, negatives=negatives)


def _validate_train_split(entries: list[dict[str, object]]) -> None:
    if not entries:
        raise ValueError("OTB YOLO export produced an empty train split")
    if not any(bool(entry["has_otb_board"]) for entry in entries):
        raise ValueError("OTB YOLO export train split has no positive board labels")


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


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
