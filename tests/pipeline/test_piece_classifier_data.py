"""Tests for pipeline.overlay.piece_classifier_data — dataset generation and persistence."""


import numpy as np
import pytest
from pipeline.overlay.piece_classifier_data import (
    CLASS_NAMES,
    NUM_CLASSES,
    generate_dataset,
    load_dataset,
    save_dataset,
)


class TestGenerateDataset:
    """Validate the synthetic dataset generator produces well-formed data."""

    @pytest.fixture(scope="class")
    def dataset(self, tmp_path_factory):
        """Generate a small dataset once for all tests in this class."""
        out = tmp_path_factory.mktemp("dataset")
        images, labels = generate_dataset(
            num_samples_per_class=5,
            size=64,
            seed=99,
            output_dir=out,
        )
        return images, labels, out

    def test_shapes(self, dataset):
        images, labels, _ = dataset
        assert images.shape == (5 * NUM_CLASSES, 64, 64, 3)
        assert labels.shape == (5 * NUM_CLASSES,)

    def test_dtypes(self, dataset):
        images, labels, _ = dataset
        assert images.dtype == np.uint8
        assert labels.dtype == np.int64

    def test_all_classes_present(self, dataset):
        _, labels, _ = dataset
        for c in range(NUM_CLASSES):
            assert c in labels, f"Class {c} ({CLASS_NAMES[c]}) missing from dataset"

    def test_balanced(self, dataset):
        _, labels, _ = dataset
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        assert counts.min() >= 4, f"Imbalanced: {counts}"  # 5 per class, shuffled

    def test_images_valid(self, dataset):
        images, _, _ = dataset
        # No all-zero images
        per_image_sum = images.reshape(len(images), -1).sum(axis=1)
        assert np.all(per_image_sum > 0), "Found all-zero images"

    def test_images_three_channel(self, dataset):
        images, _, _ = dataset
        assert images.shape[-1] == 3


class TestSaveLoadRoundtrip:
    """Verify that saving and loading produces identical arrays."""

    def test_roundtrip(self, tmp_path):
        images = np.random.randint(0, 256, (26, 64, 64, 3), dtype=np.uint8)
        labels = np.array([i % NUM_CLASSES for i in range(26)], dtype=np.int64)

        save_dataset(images, labels, tmp_path, seed=1, num_samples_per_class=2, size=64)

        loaded = load_dataset(tmp_path)
        assert loaded is not None
        loaded_images, loaded_labels = loaded

        np.testing.assert_array_equal(images, loaded_images)
        np.testing.assert_array_equal(labels, loaded_labels)

    def test_metadata_written(self, tmp_path):
        images = np.zeros((13, 32, 32, 3), dtype=np.uint8)
        labels = np.arange(13, dtype=np.int64)
        save_dataset(images, labels, tmp_path, seed=42, num_samples_per_class=1, size=32)

        import json

        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["seed"] == 42
        assert meta["num_samples_per_class"] == 1
        assert meta["size"] == 32
        assert meta["total_samples"] == 13
        assert meta["num_classes"] == NUM_CLASSES
        assert meta["class_names"] == CLASS_NAMES

    def test_load_missing_dir(self, tmp_path):
        assert load_dataset(tmp_path / "nonexistent") is None
