"""Service layer for real-footage training clip inventory and batch processing."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from api.services.data import synthetic_service

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov")

_current_job: dict[str, Any] | None = None
_cancel_event = threading.Event()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def _list_local_videos() -> list[dict[str, Any]]:
    videos_root = _resolve("data/videos")
    if not videos_root.exists():
        return []

    videos: list[dict[str, Any]] = []
    for video_dir in sorted(videos_root.iterdir()):
        if not video_dir.is_dir():
            continue
        for path in sorted(video_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _VIDEO_EXTENSIONS:
                continue
            stat = path.stat()
            videos.append(
                {
                    "video_id": video_dir.name,
                    "video_path": str(path),
                    "file_size_mb": round(stat.st_size / 1024 / 1024, 1),
                    "modified_ts": stat.st_mtime,
                }
            )
            break

    return sorted(videos, key=lambda row: row["modified_ts"], reverse=True)


def _load_video_rows(video_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not video_ids:
        return {}

    from pipeline.db.connection import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id, channel_handle, title, published_at,
                       screening_status, layout_type
                FROM youtube_videos
                WHERE video_id = ANY(%s)
                """,
                (video_ids,),
            )
            cols = [d[0] for d in cur.description]
            return {row[0]: dict(zip(cols, row)) for row in cur.fetchall()}


def _load_db_clip_counts(video_ids: list[str]) -> dict[str, int]:
    if not video_ids:
        return {}

    from pipeline.db.connection import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id, COUNT(*) AS clip_count
                FROM video_clips
                WHERE video_id = ANY(%s)
                GROUP BY video_id
                """,
                (video_ids,),
            )
            return {video_id: int(count) for video_id, count in cur.fetchall()}


def _load_calibrated_channels() -> set[str]:
    from pipeline.overlay.calibration import list_calibrations

    return set(list_calibrations().keys())


def _load_existing_clip_counts(clips_dir: str | Path) -> dict[str, int]:
    from pipeline.overlay.training_dataset import infer_source_video_id

    counts: dict[str, int] = {}
    for clip_path in _resolve(clips_dir).glob("clip_*.pt"):
        video_id = infer_source_video_id(clip_path)
        counts[video_id] = counts.get(video_id, 0) + 1
    return counts


def _sort_videos(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float, str]:
        published = row.get("published_at")
        published_ts = published.timestamp() if published is not None else 0.0
        return (published_ts, row.get("modified_ts", 0.0), row["video_id"])

    return sorted(videos, key=key, reverse=True)


def _blocker_for_video(
    *,
    video_row: dict[str, Any] | None,
    existing_clip_count: int,
    has_channel_calibration: bool,
    db_clip_count: int,
) -> str | None:
    if existing_clip_count > 0:
        return "already_processed"
    if video_row is None:
        return "missing_db_row"
    if video_row.get("screening_status") != "approved":
        return "not_approved"
    if not video_row.get("channel_handle"):
        return "missing_channel_handle"
    if db_clip_count <= 0 and not has_channel_calibration:
        return "missing_calibration"
    return None


def get_overview(
    clips_dir: str = "data/argus/train_real",
    *,
    limit: int = 100,
) -> dict[str, Any]:
    local_videos = _list_local_videos()
    selected_videos = local_videos[:limit]
    video_ids = [row["video_id"] for row in selected_videos]

    video_rows = _load_video_rows(video_ids)
    db_clip_counts = _load_db_clip_counts(video_ids)
    calibrated_channels = _load_calibrated_channels()
    existing_clip_counts = _load_existing_clip_counts(clips_dir)
    clip_stats = synthetic_service.get_clip_stats(clips_dir)

    videos: list[dict[str, Any]] = []
    for local in selected_videos:
        video_row = video_rows.get(local["video_id"])
        channel_handle = video_row.get("channel_handle") if video_row is not None else None
        existing_clip_count = existing_clip_counts.get(local["video_id"], 0)
        db_clip_count = db_clip_counts.get(local["video_id"], 0)
        has_channel_calibration = bool(channel_handle and channel_handle in calibrated_channels)
        blocker = _blocker_for_video(
            video_row=video_row,
            existing_clip_count=existing_clip_count,
            has_channel_calibration=has_channel_calibration,
            db_clip_count=db_clip_count,
        )
        ready = blocker is None
        videos.append(
            {
                "video_id": local["video_id"],
                "video_path": local["video_path"],
                "file_size_mb": local["file_size_mb"],
                "modified_ts": local["modified_ts"],
                "title": video_row.get("title") if video_row is not None else None,
                "channel_handle": channel_handle,
                "published_at": video_row.get("published_at") if video_row is not None else None,
                "screening_status": video_row.get("screening_status") if video_row is not None else None,
                "layout_type": video_row.get("layout_type") if video_row is not None else None,
                "existing_clip_count": existing_clip_count,
                "db_clip_count": db_clip_count,
                "has_channel_calibration": has_channel_calibration,
                "ready": ready,
                "blocker": blocker,
            }
        )

    videos = _sort_videos(videos)
    ready_count = sum(1 for row in videos if row["ready"])
    processed_count = sum(1 for row in videos if row["blocker"] == "already_processed")
    blocked_count = len(videos) - ready_count - processed_count

    return {
        "clips_dir": str(_resolve(clips_dir)),
        "clip_stats": clip_stats,
        "source_video_count": len(existing_clip_counts),
        "local_video_count": len(videos),
        "ready_video_count": ready_count,
        "processed_video_count": processed_count,
        "blocked_video_count": blocked_count,
        "videos": videos,
    }


def get_processing_status() -> dict[str, Any]:
    if _current_job is None:
        return {"status": "idle"}
    return {
        "job_id": _current_job["job_id"],
        "status": _current_job["status"],
        "requested_limit": _current_job["requested_limit"],
        "completed_videos": _current_job["completed_videos"],
        "total_videos": _current_job["total_videos"],
        "generated_clips": _current_job["generated_clips"],
        "current_video_id": _current_job.get("current_video_id"),
        "current_video_title": _current_job.get("current_video_title"),
        "clips_dir": _current_job["clips_dir"],
        "results": list(_current_job["results"]),
        "error": _current_job.get("error"),
    }


def start_processing(
    *,
    limit: int = 10,
    clips_dir: str = "data/argus/train_real",
    min_moves: int = 5,
) -> dict[str, Any]:
    global _current_job, _cancel_event

    if _current_job is not None and _current_job.get("status") == "running":
        raise ValueError("Real-video processing already in progress. Stop it first.")

    overview = get_overview(clips_dir, limit=5000)
    candidates = [row for row in overview["videos"] if row["ready"]][:limit]
    if not candidates:
        raise ValueError("No eligible local videos found to process.")

    import uuid

    job_id = uuid.uuid4().hex[:12]
    _cancel_event = threading.Event()
    _current_job = {
        "job_id": job_id,
        "status": "running",
        "requested_limit": limit,
        "completed_videos": 0,
        "total_videos": len(candidates),
        "generated_clips": 0,
        "current_video_id": None,
        "current_video_title": None,
        "clips_dir": str(_resolve(clips_dir)),
        "results": [],
        "error": None,
    }

    thread = threading.Thread(
        target=_run_processing,
        args=(job_id, candidates, str(_resolve(clips_dir)), min_moves),
        daemon=True,
    )
    thread.start()
    return get_processing_status()


def _run_processing(
    job_id: str,
    candidates: list[dict[str, Any]],
    clips_dir: str,
    min_moves: int,
) -> None:
    global _current_job

    from pipeline.overlay.overlay_clip_generator import generate_from_video

    try:
        for candidate in candidates:
            if _cancel_event.is_set():
                break

            if _current_job is None or _current_job.get("job_id") != job_id:
                return

            _current_job["current_video_id"] = candidate["video_id"]
            _current_job["current_video_title"] = candidate.get("title")

            try:
                results = generate_from_video(
                    candidate["video_path"],
                    channel_handle=candidate["channel_handle"],
                    output_dir=clips_dir,
                    min_moves_per_segment=min_moves,
                )
                status = "generated" if results else "no_clips"
                _current_job["generated_clips"] += len(results)
                _current_job["results"].append(
                    {
                        "video_id": candidate["video_id"],
                        "title": candidate.get("title"),
                        "status": status,
                        "generated_clip_count": len(results),
                        "error": None,
                    }
                )
            except Exception as e:  # pragma: no cover - exercised by tests via mocks
                _current_job["results"].append(
                    {
                        "video_id": candidate["video_id"],
                        "title": candidate.get("title"),
                        "status": "failed",
                        "generated_clip_count": 0,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

            _current_job["completed_videos"] += 1

        if _current_job is not None and _current_job.get("job_id") == job_id:
            _current_job["status"] = "stopped" if _cancel_event.is_set() else "done"
            _current_job["current_video_id"] = None
            _current_job["current_video_title"] = None
    except Exception as e:  # pragma: no cover - defensive guard
        if _current_job is not None and _current_job.get("job_id") == job_id:
            _current_job["status"] = "failed"
            _current_job["error"] = f"{type(e).__name__}: {e}"
            _current_job["current_video_id"] = None
            _current_job["current_video_title"] = None


def stop_processing() -> dict[str, Any]:
    if _current_job is None or _current_job.get("status") != "running":
        return {"status": "no_job_running"}

    _cancel_event.set()
    return get_processing_status()
