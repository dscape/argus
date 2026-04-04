"""Tests for pipeline.download.video_downloader.get_video_path."""

import pytest

yt_dlp = pytest.importorskip("yt_dlp", reason="yt-dlp not installed")

from pipeline.download.video_downloader import get_video_path  # noqa: E402


class TestGetVideoPath:
    """Test the video path resolution function."""

    def test_returns_path_when_file_exists(self, tmp_path, monkeypatch):
        video_dir = tmp_path / "abc123"
        video_dir.mkdir()
        video_file = video_dir / "abc123.mp4"
        video_file.write_text("fake video")

        monkeypatch.setattr("pipeline.paths.VIDEOS_DIR", tmp_path)
        result = get_video_path("abc123")
        assert result is not None
        assert result.endswith("abc123.mp4")

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pipeline.paths.VIDEOS_DIR", tmp_path)
        result = get_video_path("nonexistent")
        assert result is None

    def test_finds_video_in_directory(self, tmp_path, monkeypatch):
        video_dir = tmp_path / "xyz789"
        video_dir.mkdir()
        video_file = video_dir / "xyz789.mp4"
        video_file.write_text("fake video")

        monkeypatch.setattr("pipeline.paths.VIDEOS_DIR", tmp_path)
        result = get_video_path("xyz789")
        assert result is not None

    def test_no_video_file_in_directory(self, tmp_path, monkeypatch):
        video_dir = tmp_path / "test"
        video_dir.mkdir()
        # directory exists but no video file inside

        monkeypatch.setattr("pipeline.paths.VIDEOS_DIR", tmp_path)
        result = get_video_path("test")
        assert result is None
