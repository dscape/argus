"""Tests for the generation service start/stop/status lifecycle."""

from unittest.mock import patch

import pytest
from api.services import generation_service


@pytest.fixture(autouse=True)
def reset_service():
    """Reset the service state between tests."""
    generation_service._current_job = None
    generation_service._cancel_event.clear()
    yield
    generation_service._current_job = None
    generation_service._cancel_event.clear()


class TestGetStatus:
    def test_idle_by_default(self):
        status = generation_service.get_status()
        assert status["status"] == "idle"


class TestStartGeneration:
    @patch("api.services.generation_service._run_generation")
    def test_start_returns_running(self, mock_run):
        status = generation_service.start_generation(num_clips=10)
        assert status["status"] == "running"
        assert status["num_clips"] == 10
        assert status["completed"] == 0

    @patch("api.services.generation_service._run_generation")
    def test_cannot_start_while_running(self, mock_run):
        generation_service.start_generation(num_clips=10)
        with pytest.raises(ValueError, match="already in progress"):
            generation_service.start_generation(num_clips=5)


class TestStopGeneration:
    def test_stop_when_no_job(self):
        result = generation_service.stop_generation()
        assert result["status"] == "no_job_running"

    @patch("api.services.generation_service._run_generation")
    def test_stop_sets_cancel_event(self, mock_run):
        generation_service.start_generation(num_clips=10)
        generation_service.stop_generation()
        assert generation_service._cancel_event.is_set()


class TestRunGeneration:
    @patch("argus.datagen.synth.generate_dataset")
    def test_successful_run_sets_done(self, mock_gen):
        mock_gen.return_value = []
        generation_service._current_job = {
            "job_id": "test123",
            "status": "running",
            "num_clips": 5,
            "completed": 0,
            "output_dir": "data/train",
            "error": None,
        }
        generation_service._cancel_event.clear()

        generation_service._run_generation(
            "test123", 5, "/tmp/test_gen", 224, 16, 4, 42, "training"
        )

        assert generation_service._current_job["status"] == "done"

    @patch("argus.datagen.synth.generate_dataset", side_effect=RuntimeError("blender not found"))
    def test_failed_run_sets_failed(self, mock_gen):
        generation_service._current_job = {
            "job_id": "test456",
            "status": "running",
            "num_clips": 5,
            "completed": 0,
            "output_dir": "data/train",
            "error": None,
        }
        generation_service._cancel_event.clear()

        generation_service._run_generation(
            "test456", 5, "/tmp/test_gen", 224, 16, 4, 42, "training"
        )

        assert generation_service._current_job["status"] == "failed"
        assert "blender not found" in generation_service._current_job["error"]
