"""Tests for committed runtime weight preflight checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipeline.runtime_assets import (
    RuntimeAsset,
    ensure_default_runtime_assets,
    ensure_runtime_asset,
    is_git_lfs_pointer,
)


def test_is_git_lfs_pointer_detects_pointer_file(tmp_path: Path) -> None:
    path = tmp_path / "best.onnx"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 123\n")

    assert is_git_lfs_pointer(path) is True


def test_ensure_runtime_asset_rejects_pointer_file(tmp_path: Path) -> None:
    path = tmp_path / "best.onnx"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 123\n")

    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        ensure_runtime_asset(RuntimeAsset(name="test weights", path=path))


def test_default_runtime_assets_are_real_blobs() -> None:
    validated = ensure_default_runtime_assets()

    assert validated
    assert all(path.exists() for path in validated)
    assert all(not is_git_lfs_pointer(path) for path in validated)
