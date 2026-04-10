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


class TestSplitClipsCommand:
    """Test the split-clips CLI command."""

    def test_reports_split_counts(self, monkeypatch, capsys):
        import pipeline.overlay.training_dataset as training_dataset

        monkeypatch.setattr(
            training_dataset,
            "export_training_dataset",
            lambda clips_dir, out_dir, val_fraction=0.2, seed=42, link_mode="hardlink": {
                "splits": {
                    "train": [{"clip": "train/clip_a.pt", "source_video_id": "vidA"}],
                    "val": [{"clip": "val/clip_b.pt", "source_video_id": "vidB"}],
                }
            },
        )

        cli.cmd_split_clips(
            SimpleNamespace(
                clips_dir="data/argus/train_real",
                out_dir="data/argus/training_dataset",
                val_fraction=0.2,
                seed=42,
                copy=False,
            )
        )

        out = capsys.readouterr().out
        assert "Prepared dataset" in out
        assert "Train clips: 1" in out
        assert "Val clips:   1" in out


class TestRealDataOverviewCommand:
    """Test the real-data-overview CLI command."""

    def test_prints_summary_and_blockers(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli,
            "_get_real_data_overview",
            lambda clips_dir, max_file_size_mb=200.0, limit=100: {
                "clips_dir": "/tmp/train_real",
                "local_video_count": 3,
                "ready_video_count": 1,
                "processed_video_count": 1,
                "blocked_video_count": 1,
                "source_video_count": 1,
                "videos": [
                    {
                        "video_id": "READYVIDEO1",
                        "title": "Ready video",
                        "channel_handle": "@ready",
                        "db_clip_count": 0,
                        "existing_clip_count": 0,
                        "ready": True,
                        "blocker": None,
                        "published_at": None,
                    },
                    {
                        "video_id": "BLOCKVIDEO1",
                        "title": "Blocked video",
                        "channel_handle": "@blocked",
                        "db_clip_count": 0,
                        "existing_clip_count": 0,
                        "ready": False,
                        "blocker": "missing_calibration",
                        "published_at": None,
                    },
                    {
                        "video_id": "DONEVIDEO11",
                        "title": "Done video",
                        "channel_handle": "@done",
                        "db_clip_count": 1,
                        "existing_clip_count": 2,
                        "ready": False,
                        "blocker": "already_processed",
                        "published_at": None,
                    },
                ],
            },
        )

        cli.cmd_real_data_overview(
            SimpleNamespace(
                clips_dir="data/argus/train_real",
                limit=100,
                max_file_size_mb=200.0,
                json=False,
            )
        )

        out = capsys.readouterr().out
        assert "Local videos:     3" in out
        assert "Ready videos:     1" in out
        assert "missing_calibration: 1" in out
        assert "READYVIDEO1  [ready]" in out


class TestRealDataProcessCommand:
    """Test the real-data-process CLI command."""

    def test_processes_top_ready_videos(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli,
            "_get_real_data_overview",
            lambda clips_dir, max_file_size_mb=200.0, limit=5000: {
                "videos": [
                    {
                        "video_id": "READYVIDEO1",
                        "video_path": "/tmp/READYVIDEO1.mp4",
                        "channel_handle": "@ready",
                        "ready": True,
                    },
                    {
                        "video_id": "READYVIDEO2",
                        "video_path": "/tmp/READYVIDEO2.mp4",
                        "channel_handle": "@ready",
                        "ready": True,
                    },
                    {
                        "video_id": "BLOCKVIDEO1",
                        "video_path": "/tmp/BLOCKVIDEO1.mp4",
                        "channel_handle": "@blocked",
                        "ready": False,
                    },
                ]
            },
        )
        monkeypatch.setattr(cli, "_resolve_project_path", lambda path: "/tmp/train_real")

        import pipeline.overlay.overlay_clip_generator as clip_generator

        calls: list[tuple[str, str, str, int]] = []

        monkeypatch.setattr(
            clip_generator,
            "generate_from_video",
            lambda video_path, channel_handle, output_dir, min_moves_per_segment=5: (
                calls.append((video_path, channel_handle, output_dir, min_moves_per_segment))
                or [{"filepath": f"{output_dir}/clip.pt", "num_moves": 7, "num_frames": 120}]
            ),
        )

        cli.cmd_real_data_process(
            SimpleNamespace(
                clips_dir="data/argus/train_real",
                limit=2,
                min_moves=3,
                max_file_size_mb=200.0,
            )
        )

        out = capsys.readouterr().out
        assert "Processed 2 ready video(s)" in out
        assert "Generated 2 clip(s) into /tmp/train_real" in out
        assert calls == [
            ("/tmp/READYVIDEO1.mp4", "@ready", "/tmp/train_real", 3),
            ("/tmp/READYVIDEO2.mp4", "@ready", "/tmp/train_real", 3),
        ]


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
