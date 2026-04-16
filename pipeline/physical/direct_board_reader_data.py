"""Data helpers for full-image direct board readers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.physical.board_data import INPUT_SIZE, normalize_rgb_tensor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CHESSRED_CATEGORY_TO_LABEL = {
    0: 1,  # white pawn -> P
    1: 4,  # white rook -> R
    2: 2,  # white knight -> N
    3: 3,  # white bishop -> B
    4: 5,  # white queen -> Q
    5: 6,  # white king -> K
    6: 7,  # black pawn -> p
    7: 10,  # black rook -> r
    8: 8,  # black knight -> n
    9: 9,  # black bishop -> b
    10: 11,  # black queen -> q
    11: 12,  # black king -> k
    12: 0,  # empty
}


@dataclass(frozen=True)
class DirectBoardRecord:
    example_id: str
    domain: str
    split: str
    labels: tuple[int, ...]
    width: int
    height: int
    image_path: str | None = None
    sample_weight: float = 1.0
    annotation_id: str | None = None
    clip_path: str | None = None
    frame_index: int | None = None
    source_video_id: str | None = None
    source_frame_index: int | None = None
    native_image_bbox: tuple[int, int, int, int] | None = None
    corners: tuple[tuple[float, float], ...] | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["labels"] = list(self.labels)
        payload["corners"] = (
            None if self.corners is None else [list(point) for point in self.corners]
        )
        payload["native_image_bbox"] = (
            None if self.native_image_bbox is None else list(self.native_image_bbox)
        )
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DirectBoardRecord:
        raw_labels = payload.get("labels")
        if not isinstance(raw_labels, list) or len(raw_labels) != 64:
            raise ValueError("DirectBoardRecord payload must contain 64 labels")
        raw_corners = payload.get("corners")
        corners = None
        if isinstance(raw_corners, list) and len(raw_corners) == 4:
            corners = tuple((float(x), float(y)) for x, y in raw_corners)
        raw_bbox = payload.get("native_image_bbox")
        native_image_bbox = None
        if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
            native_image_bbox = tuple(int(value) for value in raw_bbox)
        raw_image_path = payload.get("image_path")
        return cls(
            example_id=str(payload["example_id"]),
            domain=str(payload["domain"]),
            split=str(payload["split"]),
            labels=tuple(int(value) for value in raw_labels),
            width=int(payload["width"]),
            height=int(payload["height"]),
            image_path=None if raw_image_path in {None, ""} else str(raw_image_path),
            sample_weight=float(payload.get("sample_weight", 1.0)),
            annotation_id=(
                None if payload.get("annotation_id") is None else str(payload.get("annotation_id"))
            ),
            clip_path=(None if payload.get("clip_path") is None else str(payload.get("clip_path"))),
            frame_index=(
                None if payload.get("frame_index") is None else int(payload.get("frame_index"))
            ),
            source_video_id=(
                None
                if payload.get("source_video_id") is None
                else str(payload.get("source_video_id"))
            ),
            source_frame_index=(
                None
                if payload.get("source_frame_index") is None
                else int(payload.get("source_frame_index"))
            ),
            native_image_bbox=native_image_bbox,
            corners=corners,
        )


class DirectBoardImageLoader:
    def __init__(self) -> None:
        self._clip_cache: dict[Path, dict[str, Any]] = {}
        self._video_capture_cache: dict[Path, cv2.VideoCapture] = {}

    def close(self) -> None:
        for capture in self._video_capture_cache.values():
            capture.release()
        self._video_capture_cache.clear()

    def __del__(self) -> None:
        self.close()

    def load_bgr(self, row: DirectBoardRecord) -> np.ndarray:
        if row.image_path is not None:
            image = cv2.imread(str((_PROJECT_ROOT / row.image_path).resolve()), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load direct-board image: {row.image_path}")
            return image
        if (
            row.source_video_id is not None
            and row.source_frame_index is not None
            and row.native_image_bbox is not None
        ):
            return self._load_native_video_crop(
                source_video_id=row.source_video_id,
                source_frame_index=row.source_frame_index,
                native_image_bbox=row.native_image_bbox,
            )
        if row.clip_path is not None and row.frame_index is not None:
            return self._load_clip_frame(clip_path=row.clip_path, frame_index=row.frame_index)
        raise ValueError(f"DirectBoardRecord has no readable source: {row.example_id}")

    def _load_clip_frame(self, *, clip_path: str, frame_index: int) -> np.ndarray:
        clip_abs_path = (_PROJECT_ROOT / clip_path).resolve()
        clip = self._clip_cache.get(clip_abs_path)
        if clip is None:
            loaded = torch.load(clip_abs_path, map_location="cpu", weights_only=False)
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid clip payload: {clip_path}")
            self._clip_cache[clip_abs_path] = loaded
            clip = loaded
        frames = clip.get("frames")
        if not isinstance(frames, torch.Tensor):
            raise ValueError(f"Clip has no frames tensor: {clip_path}")
        return _frame_tensor_to_bgr(frames[frame_index])

    def _load_native_video_crop(
        self,
        *,
        source_video_id: str,
        source_frame_index: int,
        native_image_bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        video_path = resolve_source_video_path(source_video_id)
        capture = self._video_capture_cache.get(video_path)
        if capture is None:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError(f"Failed to open source video: {video_path}")
            self._video_capture_cache[video_path] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Failed to read frame {source_frame_index} from {video_path}")
        x, y, w, h = native_image_bbox
        crop = frame[y : y + h, x : x + w].copy()
        if crop.size == 0:
            raise ValueError(
                f"Invalid native crop for {source_video_id} frame {source_frame_index}:"
                f" {native_image_bbox}"
            )
        return crop


class DirectBoardManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        image_size: int = INPUT_SIZE,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.rows = load_direct_board_records(self.manifest_path)
        self._image_loader = DirectBoardImageLoader()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = self._image_loader.load_bgr(row)
        image_tensor = preprocess_full_board_image(image, size=self.image_size)
        labels = torch.tensor(row.labels, dtype=torch.long)
        weight = torch.tensor(row.sample_weight, dtype=torch.float32)
        return image_tensor, labels, weight


def square_name_to_index(square_name: str) -> int:
    if len(square_name) != 2:
        raise ValueError(f"Expected algebraic square like 'a8', got {square_name!r}")
    file_index = ord(square_name[0].lower()) - ord("a")
    rank = int(square_name[1])
    if file_index < 0 or file_index > 7 or rank < 1 or rank > 8:
        raise ValueError(f"Invalid square name: {square_name!r}")
    row = 8 - rank
    return row * 8 + file_index


def chessred_labels_by_image_id(payload: dict[str, Any]) -> dict[int, tuple[int, ...]]:
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, dict):
        raise ValueError("Unexpected ChessReD payload structure")
    piece_annotations = annotations.get("pieces")
    if not isinstance(piece_annotations, list):
        raise ValueError("ChessReD annotations must contain a 'pieces' list")

    labels_by_image_id: dict[int, list[int]] = {
        int(image_record["id"]): [0] * 64 for image_record in images
    }
    for piece in piece_annotations:
        image_id = int(piece["image_id"])
        category_id = int(piece["category_id"])
        square_name = str(piece["chessboard_position"])
        class_id = _CHESSRED_CATEGORY_TO_LABEL[category_id]
        square_index = square_name_to_index(square_name)
        labels = labels_by_image_id[image_id]
        labels[square_index] = class_id
    return {image_id: tuple(labels) for image_id, labels in labels_by_image_id.items()}


def chessred_image_path(image_record: dict[str, Any], *, images_root: Path) -> Path:
    direct = images_root / str(image_record.get("path", ""))
    if direct.exists():
        return direct
    game_id = int(image_record["game_id"])
    fallback = images_root / str(game_id) / str(image_record["file_name"])
    if fallback.exists():
        return fallback
    raise ValueError(
        "ChessReD image is missing; tried"
        f" {direct.relative_to(_PROJECT_ROOT) if direct.is_absolute() else direct} and"
        f" {fallback.relative_to(_PROJECT_ROOT) if fallback.is_absolute() else fallback}"
    )


def preprocess_full_board_image(image_bgr: np.ndarray, *, size: int = INPUT_SIZE) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = float(size) / max(float(height), float(width), 1.0)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if max(height, width) >= size else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=interpolation)

    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    x_offset = (size - resized_width) // 2
    y_offset = (size - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized

    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    return normalize_rgb_tensor(tensor)


def load_direct_board_records(manifest_path: str | Path) -> list[DirectBoardRecord]:
    path = Path(manifest_path)
    rows: list[DirectBoardRecord] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(DirectBoardRecord.from_json(json.loads(line)))
    return rows


def write_direct_board_records(path: str | Path, rows: list[DirectBoardRecord]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(json.dumps(row.to_json(), sort_keys=True) for row in rows) + "\n"
    )


def resolve_source_video_path(source_video_id: str) -> Path:
    video_path = _PROJECT_ROOT / "data" / "videos" / source_video_id / f"{source_video_id}.mp4"
    if not video_path.exists():
        raise ValueError(f"Source video is missing for {source_video_id}: {video_path}")
    return video_path


def _frame_tensor_to_bgr(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected a CHW or HWC frame tensor, got {tuple(frame.shape)}")
    if frame.shape[0] == 3:
        chw = frame
    elif frame.shape[-1] == 3:
        chw = frame.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB frame tensor, got {tuple(frame.shape)}")

    rgb = chw.to(torch.float32)
    if float(rgb.max().item()) <= 1.0:
        rgb = rgb * 255.0
    rgb_uint8 = rgb.clamp(0.0, 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
