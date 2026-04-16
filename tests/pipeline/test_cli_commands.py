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


class TestPhysicalSplitClipsCommand:
    """Test the physical-split-clips CLI command."""

    def test_reports_split_counts_and_exclusions(self, monkeypatch, capsys):
        import pipeline.physical.training_dataset as training_dataset

        monkeypatch.setattr(
            training_dataset,
            "export_physical_training_dataset",
            lambda clips_dir, out_dir, val_fraction=0.2, seed=42, link_mode="hardlink": {
                "excluded_source_video_ids": ["vidHoldoutA", "vidHoldoutB"],
                "excluded_clip_count": 3,
                "splits": {
                    "train": [{"clip": "train/clip_a.pt", "source_video_id": "vidA"}],
                    "val": [{"clip": "val/clip_b.pt", "source_video_id": "vidB"}],
                },
            },
        )

        cli.cmd_physical_split_clips(
            SimpleNamespace(
                clips_dir="data/argus/train_real",
                out_dir="data/physical/training_dataset",
                val_fraction=0.2,
                seed=42,
                copy=False,
            )
        )

        out = capsys.readouterr().out
        assert "Prepared physical dataset" in out
        assert "Train clips:         1" in out
        assert "Val clips:           1" in out
        assert "Excluded videos:     2" in out
        assert "Excluded clip count: 3" in out


class TestPhysicalBoardFailureStudyCommand:
    """Test the physical-board-failure-study CLI command."""

    def test_prints_failure_study_summary(self, monkeypatch, capsys):
        import pipeline.physical.board_tracker_failure_study as failure_study

        monkeypatch.setattr(
            failure_study,
            "load_config_from_eval_report",
            lambda path: failure_study.TrackerFailureStudyConfig(
                tracker_mode="lookahead",
                observation_input="rectified_board",
            ),
        )
        monkeypatch.setattr(
            failure_study,
            "create_tracker_failure_study",
            lambda **kwargs: {
                "total_failures": 42,
                "selected_failures": 10,
                "manifest": "outputs/physical_board_failure_study/manifest.json",
                "manual_buckets_csv": "outputs/physical_board_failure_study/manual_buckets.csv",
                "contact_sheet": "outputs/physical_board_failure_study/contact_sheet.png",
                "summary_path": "outputs/physical_board_failure_study/summary.json",
            },
        )

        cli.cmd_physical_board_failure_study(
            SimpleNamespace(
                eval_report="outputs/eval.json",
                observation_input=None,
                temporal_mode=None,
                temporal_ema_alpha=None,
                tracker_mode=None,
                move_accept_threshold=None,
                move_accept_margin=None,
                lookahead_window=None,
                lookahead_margin=None,
                weights_path=None,
                limit=10,
                sample_mode="round_robin",
                top_legal_candidates=5,
                panel_size=240,
                device="cpu",
                output_dir="outputs/physical_board_failure_study",
            )
        )

        out = capsys.readouterr().out
        assert "Built physical board failure study" in out
        assert "Total failures:     42" in out
        assert "Selected failures:  10" in out
        assert "Manifest:           outputs/physical_board_failure_study/manifest.json" in out
        assert "Contact sheet:      outputs/physical_board_failure_study/contact_sheet.png" in out


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


class TestRealDataAuditCommand:
    """Test the real-data-audit CLI command."""

    def test_prints_audit_summary(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli,
            "_get_real_data_overview",
            lambda clips_dir, max_file_size_mb=200.0, limit=5000: {
                "videos": [
                    {
                        "video_id": "READYVIDEO1",
                        "video_path": "/tmp/READYVIDEO1.mp4",
                        "channel_handle": "@ready",
                        "title": "Ready video",
                        "existing_clip_count": 0,
                        "ready": True,
                    }
                ]
            },
        )

        import pipeline.overlay.real_video_audit as real_video_audit

        monkeypatch.setattr(
            real_video_audit,
            "audit_video_generation",
            lambda video_id, video_path, channel_handle, output_dir, min_moves_per_segment: {
                "video_id": video_id,
                "generated_clip_count": 0,
                "failure_reason": "illegal_jump_fragmentation",
                "max_segment_moves": 4,
                "hard_cut_count": 1,
                "illegal_jump_count": 6,
                "clip_reports": [
                    {
                        "clip_label": video_id,
                        "failure_reason": "illegal_jump_fragmentation",
                        "segment_count": 2,
                        "max_segment_moves": 4,
                        "hard_cut_count": 1,
                        "illegal_jump_count": 6,
                    }
                ],
            },
        )

        cli.cmd_real_data_audit(
            SimpleNamespace(
                video_id=None,
                clips_dir="data/argus/train_real",
                limit=10,
                min_moves=5,
                max_file_size_mb=200.0,
                json=False,
            )
        )

        out = capsys.readouterr().out
        assert "Videos audited:   1" in out
        assert "illegal_jump_fragmentation: 1" in out
        assert "READYVIDEO1  [illegal_jump_fragmentation]" in out
        assert "segments=2  max_moves=4  hard_cuts=1  illegal=6" in out

    def test_json_output(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli,
            "_get_real_data_overview",
            lambda clips_dir, max_file_size_mb=200.0, limit=5000: {
                "videos": [
                    {
                        "video_id": "READYVIDEO1",
                        "video_path": "/tmp/READYVIDEO1.mp4",
                        "channel_handle": "@ready",
                        "title": "Ready video",
                        "existing_clip_count": 0,
                        "ready": True,
                    }
                ]
            },
        )

        import pipeline.overlay.real_video_audit as real_video_audit

        monkeypatch.setattr(
            real_video_audit,
            "audit_video_generation",
            lambda video_id, video_path, channel_handle, output_dir, min_moves_per_segment: {
                "video_id": video_id,
                "generated_clip_count": 1,
                "failure_reason": None,
                "max_segment_moves": 8,
                "hard_cut_count": 0,
                "illegal_jump_count": 0,
                "clip_reports": [],
            },
        )

        cli.cmd_real_data_audit(
            SimpleNamespace(
                video_id="READYVIDEO1",
                clips_dir="data/argus/train_real",
                limit=10,
                min_moves=5,
                max_file_size_mb=200.0,
                json=True,
            )
        )

        out = capsys.readouterr().out
        assert '"video_count": 1' in out
        assert '"would_generate_clips": 1' in out


class TestReferencePgnBenchmarkCommand:
    """Test the reference-pgn-benchmark CLI command."""

    def test_prints_summary(self, monkeypatch, capsys):
        import pipeline.overlay.reference_pgn_benchmark as reference_pgn_benchmark

        monkeypatch.setattr(
            reference_pgn_benchmark,
            "benchmark_reference_game",
            lambda pgn_path, video_id, clips_dir: {
                "video_id": video_id,
                "white": "White",
                "black": "Black",
                "result": "1-0",
                "reference_plies": 10,
                "coverage_plies": 6,
                "coverage_ratio": 0.6,
                "coverage_runs": [{"start_ply": 2, "end_ply": 7, "plies": 6}],
                "gaps": [{"start_ply": 0, "end_ply": 1, "plies": 2}],
                "clips": [
                    {
                        "clip_path": "data/argus/train_real/clip_overlay_demo123_0.pt",
                        "segment_start_time_seconds": 12.0,
                        "segment_end_time_seconds": 30.0,
                        "clip_plies": 6,
                        "exact_match_offsets": [2],
                        "longest_prefix_start_ply": 2,
                        "longest_prefix_plies": 6,
                    }
                ],
            },
        )

        cli.cmd_reference_pgn_benchmark(
            SimpleNamespace(
                pgn="outputs/reference/chesscom/demo123.pgn",
                video_id="demo123",
                clips_dir="data/argus/train_real",
                json=False,
            )
        )

        out = capsys.readouterr().out
        assert "Video:           demo123" in out
        assert "Coverage:        6/10 (60.0%)" in out
        assert "clip_overlay_demo123_0.pt" in out
        assert "2-7 (6 plies)" in out

    def test_json_output(self, monkeypatch, capsys):
        import pipeline.overlay.reference_pgn_benchmark as reference_pgn_benchmark

        monkeypatch.setattr(
            reference_pgn_benchmark,
            "benchmark_reference_game",
            lambda pgn_path, video_id, clips_dir: {
                "video_id": video_id,
                "coverage_ratio": 0.5,
                "coverage_runs": [],
                "gaps": [],
                "clips": [],
                "white": "White",
                "black": "Black",
                "result": "1/2-1/2",
                "reference_plies": 0,
                "coverage_plies": 0,
            },
        )

        cli.cmd_reference_pgn_benchmark(
            SimpleNamespace(
                pgn="outputs/reference/chesscom/demo123.pgn",
                video_id="demo123",
                clips_dir="data/argus/train_real",
                json=True,
            )
        )

        out = capsys.readouterr().out
        assert '"video_id": "demo123"' in out
        assert '"coverage_ratio": 0.5' in out


class TestAutoSegmentVideoCommand:
    """Test the auto-segment-video CLI command."""

    def test_prints_summary(self, monkeypatch, capsys):
        import pipeline.overlay.clip_workflow as clip_workflow

        monkeypatch.setattr(
            clip_workflow,
            "auto_segment_video",
            lambda video_id, sample_interval_sec, replace_existing: {
                "segments": [
                    {
                        "clip_id": 12,
                        "start_time": 5.0,
                        "end_time": 42.5,
                        "score": 0.91,
                        "overlay_bbox": [10, 20, 300, 300],
                    }
                ],
                "gaps": [{"start_time": 0.0, "end_time": 5.0}],
                "video_resolution": [1920, 1080],
                "total_frames_sampled": 48,
                "processing_time_sec": 2.3,
            },
        )

        cli.cmd_auto_segment_video(
            SimpleNamespace(
                video_id="demo123",
                sample_interval_sec=30.0,
                replace_existing=False,
                json=False,
            )
        )

        out = capsys.readouterr().out
        assert "Video:            demo123" in out
        assert "Segments created: 1" in out
        assert "clip_id=12" in out
        assert "overlay=[10, 20, 300, 300]" in out


class TestAutoCalibrateClipCommand:
    """Test the auto-calibrate-clip CLI command."""

    def test_prints_applied_summary(self, monkeypatch, capsys):
        import pipeline.overlay.clip_workflow as clip_workflow

        monkeypatch.setattr(
            clip_workflow,
            "auto_calibrate_clip",
            lambda video_id, clip_id: {
                "clip_id": clip_id,
                "applied": True,
                "proposal": {
                    "overlay_bbox": [1, 2, 3, 4],
                    "camera_bbox": [5, 6, 7, 8],
                    "board_theme": "lichess_default",
                    "theme_confidence": 0.98,
                    "board_flipped": False,
                    "orientation_confidence": 0.95,
                    "ref_resolution": [1920, 1080],
                },
                "failure_reason": None,
                "detected_overlay_bbox": [1, 2, 3, 4],
                "preview_frame_b64": None,
            },
        )

        cli.cmd_auto_calibrate_clip(
            SimpleNamespace(
                video_id="demo123",
                clip_id=55,
                json=False,
            )
        )

        out = capsys.readouterr().out
        assert "Clip:             55" in out
        assert "Applied:          True" in out
        assert "Overlay bbox:     [1, 2, 3, 4]" in out
        assert "Camera bbox:      [5, 6, 7, 8]" in out

    def test_prints_failure_summary(self, monkeypatch, capsys):
        import pipeline.overlay.clip_workflow as clip_workflow

        monkeypatch.setattr(
            clip_workflow,
            "auto_calibrate_clip",
            lambda video_id, clip_id: {
                "clip_id": clip_id,
                "applied": False,
                "proposal": None,
                "failure_reason": "overlay_not_found",
                "detected_overlay_bbox": None,
                "preview_frame_b64": None,
            },
        )

        cli.cmd_auto_calibrate_clip(
            SimpleNamespace(
                video_id="demo123",
                clip_id=55,
                json=False,
            )
        )

        out = capsys.readouterr().out
        assert "Applied:          False" in out
        assert "Failure reason:   overlay_not_found" in out


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
