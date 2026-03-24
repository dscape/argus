"""Tests for new CLI commands: smoke-test, inspect-calibration, ai-extract-status."""

import subprocess
import sys


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


class TestAiExtractStatusCommand:
    """Test the ai-extract-status CLI command."""

    def test_runs_without_error(self):
        """Should run even with empty/missing cache dir."""
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.cli", "ai-extract-status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "AI feature extraction progress" in result.stdout
