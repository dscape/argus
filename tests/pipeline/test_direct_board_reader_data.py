from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from pipeline.physical.direct_board_reader_data import (
    DirectBoardImageLoader,
    DirectBoardManifestDataset,
    DirectBoardRecord,
    chessred_labels_by_image_id,
    preprocess_full_board_image,
    square_name_to_index,
    write_direct_board_records,
)


def test_square_name_to_index_uses_a8_origin() -> None:
    assert square_name_to_index("a8") == 0
    assert square_name_to_index("h8") == 7
    assert square_name_to_index("a1") == 56
    assert square_name_to_index("h1") == 63


def test_chessred_labels_by_image_id_maps_piece_categories_to_board_labels() -> None:
    payload = {
        "images": [
            {"id": 1},
            {"id": 2},
        ],
        "annotations": {
            "pieces": [
                {
                    "image_id": 1,
                    "category_id": 0,
                    "chessboard_position": "a2",
                },
                {
                    "image_id": 1,
                    "category_id": 11,
                    "chessboard_position": "h8",
                },
                {
                    "image_id": 2,
                    "category_id": 4,
                    "chessboard_position": "d5",
                },
            ]
        },
    }

    labels = chessred_labels_by_image_id(payload)

    assert labels[1][square_name_to_index("a2")] == 1
    assert labels[1][square_name_to_index("h8")] == 12
    assert labels[2][square_name_to_index("d5")] == 5
    assert labels[1][square_name_to_index("a8")] == 0


def test_preprocess_full_board_image_letterboxes_to_square_tensor() -> None:
    image = np.full((120, 240, 3), 127, dtype=np.uint8)
    cv2.rectangle(image, (60, 30), (180, 90), (255, 255, 255), thickness=-1)

    tensor = preprocess_full_board_image(image, size=224)

    assert tuple(tensor.shape) == (3, 224, 224)


def test_direct_board_record_json_round_trip() -> None:
    record = DirectBoardRecord(
        example_id="demo:1",
        domain="demo",
        split="train",
        image_path="data/demo.jpg",
        labels=tuple([0] * 64),
        width=320,
        height=240,
        sample_weight=3.0,
        corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        native_image_bbox=(1, 2, 3, 4),
        source_frame_index=9,
    )

    restored = DirectBoardRecord.from_json(json.loads(json.dumps(record.to_json())))

    assert restored == record


def test_manifest_dataset_reads_synthetic_clip_records(tmp_path) -> None:
    import pipeline.physical.direct_board_reader_data as data_module

    project_root = tmp_path
    clip_path = project_root / "data" / "argus" / "train" / "clip_000000.pt"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frames": torch.zeros((1, 3, 32, 32), dtype=torch.float32),
            "fens": ["8/8/8/8/8/8/8/K6k w - - 0 1"],
            "board_flipped": False,
        },
        clip_path,
    )
    manifest_path = project_root / "data" / "direct_board_reader_dataset" / "train_manifest.jsonl"
    write_direct_board_records(
        manifest_path,
        [
            DirectBoardRecord(
                example_id="synthetic:test:0000",
                domain="synthetic",
                split="train",
                labels=tuple([0] * 64),
                width=32,
                height=32,
                clip_path="data/argus/train/clip_000000.pt",
                frame_index=0,
            )
        ],
    )

    original_root = data_module._PROJECT_ROOT
    data_module._PROJECT_ROOT = project_root
    try:
        dataset = DirectBoardManifestDataset(manifest_path=manifest_path, image_size=32)
        image, labels, weight = dataset[0]
    finally:
        data_module._PROJECT_ROOT = original_root

    assert image.shape == (3, 32, 32)
    assert labels.shape == (64,)
    assert float(weight.item()) == 1.0


def test_image_loader_reads_native_video_crop(tmp_path) -> None:
    import pipeline.physical.direct_board_reader_data as data_module

    project_root = tmp_path
    video_path = project_root / "data" / "videos" / "demo" / "demo.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[5:15, 7:17] = 255
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0,
        (32, 24),
    )
    writer.write(frame)
    writer.release()

    original_root = data_module._PROJECT_ROOT
    data_module._PROJECT_ROOT = project_root
    try:
        loader = DirectBoardImageLoader()
        crop = loader.load_bgr(
            DirectBoardRecord(
                example_id="physical:demo",
                domain="physical",
                split="train",
                labels=tuple([0] * 64),
                width=10,
                height=10,
                source_video_id="demo",
                source_frame_index=0,
                native_image_bbox=(7, 5, 10, 10),
            )
        )
        loader.close()
    finally:
        data_module._PROJECT_ROOT = original_root

    assert crop.shape == (10, 10, 3)
    assert int(crop.mean()) > 200
