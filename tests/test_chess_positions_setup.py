from pathlib import Path

from pipeline.setup import chess_positions


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def test_dataset_status_requires_populated_train_and_val(tmp_path: Path) -> None:
    overlay_dir = tmp_path / "overlay"
    (overlay_dir / "train").mkdir(parents=True)
    (overlay_dir / "val").mkdir(parents=True)

    status = chess_positions.get_dataset_status(overlay_dir)
    assert not status.ready
    assert status.missing_splits == ["data/overlay/train", "data/overlay/val"]

    _touch_image(overlay_dir / "train" / "board-1.jpg")
    status = chess_positions.get_dataset_status(overlay_dir)
    assert not status.ready
    assert status.missing_splits == ["data/overlay/val"]

    _touch_image(overlay_dir / "val" / "board-2.png")
    status = chess_positions.get_dataset_status(overlay_dir)
    assert status.ready
    assert status.missing_splits == []


def test_install_downloaded_dataset_maps_test_split_to_val(tmp_path: Path) -> None:
    download_dir = tmp_path / "download"
    overlay_dir = tmp_path / "overlay"

    _touch_image(download_dir / "train" / "train-board.jpg")
    _touch_image(download_dir / "test" / "test-board.jpg")

    status = chess_positions.install_downloaded_dataset(download_dir, overlay_dir)

    assert status.ready
    assert (overlay_dir / "train" / "train-board.jpg").exists()
    assert (overlay_dir / "val" / "test-board.jpg").exists()


def test_install_downloaded_dataset_accepts_nested_val_split(tmp_path: Path) -> None:
    download_dir = tmp_path / "download"
    overlay_dir = tmp_path / "overlay"

    _touch_image(download_dir / "dataset" / "train" / "train-board.jpg")
    _touch_image(download_dir / "dataset" / "val" / "val-board.jpg")

    status = chess_positions.install_downloaded_dataset(download_dir, overlay_dir)

    assert status.ready
    assert (overlay_dir / "train" / "train-board.jpg").exists()
    assert (overlay_dir / "val" / "val-board.jpg").exists()
