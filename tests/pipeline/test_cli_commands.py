"""Tests for selected pipeline CLI commands."""

import subprocess
import sys
from types import SimpleNamespace

import pipeline.cli as cli


class TestSmokeTestCommand:
    """Test the smoke-test CLI command."""

    def test_smoke_test_runs_successfully(self):
        """smoke-test should pass without any DB or external dependencies."""
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.cli", "smoke-test"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "All smoke tests passed" in result.stdout

    def test_smoke_test_checks_hard_cut_detection(self):
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.cli", "smoke-test"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "Hard Cut Detection" in result.stdout
        assert "PASS" in result.stdout

    def test_smoke_test_checks_ai_classifier(self):
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.cli", "smoke-test"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "AI Classifier" in result.stdout
        assert "Logits shape" in result.stdout


class TestAnalyzeVideoCommand:
    """Test the analyze-video CLI command."""

    def test_help_lists_analysis_options(self):
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.cli", "analyze-video", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--reader" in result.stdout
        assert "--scene" in result.stdout


class TestGenerateClipsCommand:
    """Test the generate-clips CLI command."""

    def test_single_video_mode_uses_new_get_video_path_signature(self, monkeypatch, capsys):
        import pipeline.db.connection as db_connection
        import pipeline.download.video_downloader as video_downloader
        import pipeline.overlay.overlay_clip_generator as clip_generator

        class FakeCursor:
            def execute(self, _query, _params):
                return None

            def fetchone(self):
                return ("@STLChessClub",)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        calls: list[str] = []

        monkeypatch.setattr(db_connection, "get_conn", lambda: FakeConn())
        monkeypatch.setattr(
            video_downloader,
            "get_video_path",
            lambda video_id: calls.append(video_id) or "/tmp/demo.mp4",
        )
        monkeypatch.setattr(
            clip_generator,
            "generate_from_video",
            lambda video_path, channel_handle, min_moves_per_segment=5: [
                {"filepath": video_path, "num_moves": 3, "num_frames": 10}
            ],
        )

        cli.cmd_generate_clips(
            SimpleNamespace(video_id="demo123", channel=None, limit=None, min_moves=5)
        )

        assert calls == ["demo123"]
        assert "Generated 1 clip(s)" in capsys.readouterr().out

    def test_batch_mode_uses_new_get_video_path_signature(self, monkeypatch, capsys):
        import pipeline.db.connection as db_connection
        import pipeline.download.video_downloader as video_downloader
        import pipeline.overlay.overlay_clip_generator as clip_generator

        class FakeCursor:
            def execute(self, _query, _params):
                return None

            def fetchall(self):
                return [("demo123", "@STLChessClub")]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        calls: list[str] = []

        monkeypatch.setattr(db_connection, "get_conn", lambda: FakeConn())
        monkeypatch.setattr(
            video_downloader,
            "get_video_path",
            lambda video_id: calls.append(video_id) or "/tmp/demo.mp4",
        )
        monkeypatch.setattr(
            clip_generator,
            "generate_from_video",
            lambda video_path, channel_handle, min_moves_per_segment=5: [
                {"filepath": video_path, "num_moves": 3, "num_frames": 10}
            ],
        )

        cli.cmd_generate_clips(
            SimpleNamespace(video_id=None, channel=None, limit=None, min_moves=5)
        )

        assert calls == ["demo123"]
        assert "Generated 1 clip(s) total" in capsys.readouterr().out


class TestInspectCalibrationCommand:
    """Test the inspect-calibration CLI command."""

    def test_requires_channel_arg(self):
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.cli", "inspect-calibration"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_missing_channel_reports_not_found(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pipeline.cli",
                "inspect-calibration",
                "--channel",
                "@NonExistentChannel12345",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "No calibration found" in result.stdout
