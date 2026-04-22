"""Service layer for managing synthetic data generation jobs."""

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Single active job (generation is resource-intensive, only one at a time)
_current_job: dict[str, Any] | None = None
_cancel_event: threading.Event = threading.Event()


def _resolve(directory: str) -> Path:
    """Resolve a directory path relative to the project root."""
    p = Path(directory)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def get_status() -> dict[str, Any]:
    """Return current generation job status."""
    if _current_job is None:
        return {"status": "idle"}
    return {
        "job_id": _current_job["job_id"],
        "status": _current_job["status"],
        "num_clips": _current_job["num_clips"],
        "completed": _current_job["completed"],
        "output_dir": _current_job["output_dir"],
        "broadcast_bias": _current_job.get("broadcast_bias"),
        "error": _current_job.get("error"),
    }


def start_generation(
    num_clips: int = 100,
    output_dir: str = "data/argus/train",
    image_size: int = 224,
    clip_length: int = 16,
    frames_per_move: int = 4,
    seed: int = 42,
    quality: str = "training",
    broadcast_bias: float = 0.0,
) -> dict[str, Any]:
    """Start a generation job in a background thread. Returns job info."""
    global _current_job, _cancel_event

    if _current_job is not None and _current_job["status"] == "running":
        raise ValueError("Generation already in progress. Stop it first.")

    import uuid

    job_id = uuid.uuid4().hex[:12]
    abs_output = str(_resolve(output_dir))

    _cancel_event = threading.Event()
    _current_job = {
        "job_id": job_id,
        "status": "running",
        "num_clips": num_clips,
        "completed": 0,
        "output_dir": output_dir,
        "error": None,
        "broadcast_bias": broadcast_bias,
    }

    thread = threading.Thread(
        target=_run_generation,
        args=(
            job_id,
            num_clips,
            abs_output,
            image_size,
            clip_length,
            frames_per_move,
            seed,
            quality,
            broadcast_bias,
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
    broadcast_bias: float,
) -> None:
    """Background thread target: run generate_dataset()."""
    global _current_job

    try:
        from argus.datagen.synth import generate_dataset

        def on_progress(completed: int, total: int) -> None:
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
            broadcast_bias=broadcast_bias,
            on_progress=on_progress,
            cancel_check=cancel_check,
        )

        if _current_job and _current_job["job_id"] == job_id:
            if _cancel_event.is_set():
                _current_job["status"] = "stopped"
            else:
                _current_job["status"] = "done"
    except Exception as e:
        logger.exception(f"Generation job {job_id} failed")
        if _current_job and _current_job["job_id"] == job_id:
            _current_job["status"] = "failed"
            _current_job["error"] = f"{type(e).__name__}: {e}"


def stop_generation() -> dict[str, Any]:
    """Stop the current generation job."""
    if _current_job is None or _current_job["status"] != "running":
        return {"status": "no_job_running"}

    _cancel_event.set()
    return get_status()
