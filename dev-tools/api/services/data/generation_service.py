"""Background job management for synthetic data generation."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_current_job: dict[str, Any] | None = None
_cancel_event = threading.Event()


def _resolve(directory: str) -> Path:
    path = Path(directory)
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


def get_status() -> dict[str, Any]:
    """Return the status of the currently running generation job."""
    if _current_job is None:
        return {"status": "idle"}

    return {
        "job_id": _current_job["job_id"],
        "status": _current_job["status"],
        "num_clips": _current_job["num_clips"],
        "completed": _current_job["completed"],
        "output_dir": _current_job["output_dir"],
        "error": _current_job.get("error"),
    }


def start_generation(
    num_clips: int = 100,
    output_dir: str = "data/train",
    image_size: int = 224,
    clip_length: int = 16,
    frames_per_move: int = 4,
    seed: int = 42,
    quality: str = "training",
) -> dict[str, Any]:
    """Start a synthetic generation job in a background thread."""
    global _current_job, _cancel_event

    if _current_job is not None and _current_job["status"] == "running":
        raise ValueError("Generation already in progress. Stop it first.")

    job_id = uuid.uuid4().hex[:12]
    absolute_output = str(_resolve(output_dir))

    _cancel_event = threading.Event()
    _current_job = {
        "job_id": job_id,
        "status": "running",
        "num_clips": num_clips,
        "completed": 0,
        "output_dir": output_dir,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_generation,
        args=(
            job_id,
            num_clips,
            absolute_output,
            image_size,
            clip_length,
            frames_per_move,
            seed,
            quality,
        ),
        daemon=True,
    )
    thread.start()

    return get_status()


def _run_generation(
    job_id: str,
    num_clips: int,
    output_dir: str,
    image_size: int,
    clip_length: int,
    frames_per_move: int,
    seed: int,
    quality: str,
) -> None:
    global _current_job

    try:
        from argus.datagen.synth import generate_dataset

        def on_progress(completed: int, _total: int) -> None:
            if _current_job and _current_job["job_id"] == job_id:
                _current_job["completed"] = completed

        def cancel_check() -> bool:
            return _cancel_event.is_set()

        generate_dataset(
            num_clips=num_clips,
            output_dir=output_dir,
            image_size=image_size,
            clip_length=clip_length,
            frames_per_move=frames_per_move,
            seed=seed,
            quality=quality,
            on_progress=on_progress,
            cancel_check=cancel_check,
        )

        if _current_job and _current_job["job_id"] == job_id:
            _current_job["status"] = "stopped" if _cancel_event.is_set() else "done"
    except Exception as error:
        logger.exception("Generation job %s failed", job_id)
        if _current_job and _current_job["job_id"] == job_id:
            _current_job["status"] = "failed"
            _current_job["error"] = f"{type(error).__name__}: {error}"


def stop_generation() -> dict[str, Any]:
    """Signal the current generation job to stop."""
    if _current_job is None or _current_job["status"] != "running":
        return {"status": "no_job_running"}

    _cancel_event.set()
    return get_status()
