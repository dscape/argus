"""Tests for pipeline.download.video_downloader.get_video_path."""

import pytest


yt_dlp = pytest.importorskip("yt_dlp", reason="yt-dlp not installed")

from pipeline.download.video_downloader import get_video_path


class TestGetVideoPath:
    """Test the video path resolution function."""

    def test_returns_path_when_file_exists(self, tmp_path):
        channel_dir = tmp_path / "STLChessClub"
        channel_dir.mkdir()
        video_file = channel_dir / "abc123.mp4"
        video_file.write_text("fake video")

        result = get_video_path("abc123", "@STLChessClub", str(tmp_path))
        assert result is not None
        assert result.endswith("abc123.mp4")

    def test_returns_none_when_missing(self, tmp_path):
        result = get_video_path("nonexistent", "@SomeChannel", str(tmp_path))
        assert result is None

    def test_strips_at_from_handle(self, tmp_path):
        channel_dir = tmp_path / "STLChessClub"
        channel_dir.mkdir()
        video_file = channel_dir / "xyz789.mp4"
        video_file.write_text("fake video")

        result = get_video_path("xyz789", "@STLChessClub", str(tmp_path))
        assert result is not None

    def test_unknown_channel_handle(self, tmp_path):
        channel_dir = tmp_path / "unknown"
        channel_dir.mkdir()
        video_file = channel_dir / "test.mp4"
        video_file.write_text("fake video")

        result = get_video_path("test", None, str(tmp_path))
        assert result is not None
