"""Crawl management endpoints — channels, videos, title scoring, AI classification."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.services import crawl_service, inspect_service, models_service

router = APIRouter()


# ── Request models ──────────────────────────────────────────


class AddChannelRequest(BaseModel):
    handle: str


class ToggleChannelRequest(BaseModel):
    enabled: bool


class UpdateStatusRequest(BaseModel):
    status: str | None = None
    layout_type: str | None = None


class BatchStatusRequest(BaseModel):
    video_ids: list[str]
    status: str


class ScreenRequest(BaseModel):
    channel_id: str | None = None
    limit: int = 500


class AutoClassifyRequest(BaseModel):
    channel_id: str | None = None
    limit: int = 200


class ClassifyRequest(BaseModel):
    video_ids: list[str]


class AiScreenRequest(BaseModel):
    video_ids: list[str]
    threshold: float = Field(default=0.90, ge=0.0, le=1.0)


class BatchInspectRequest(BaseModel):
    video_ids: list[str]


class UndoAutoRejectRequest(BaseModel):
    video_ids: list[str]


# ── Channel endpoints ──────────────────────────────────────


@router.get("/channels")
async def list_channels(screened_only: bool = Query(False)):
    return await run_in_threadpool(crawl_service.list_channels, screened_only)


@router.post("/channels")
async def add_channel(body: AddChannelRequest):
    try:
        return await run_in_threadpool(crawl_service.add_channel, body.handle)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/channels/{channel_id}")
async def get_channel_detail(channel_id: str):
    result = await run_in_threadpool(crawl_service.get_channel_detail, channel_id)
    if result is None:
        raise HTTPException(404, f"Channel {channel_id} not found")
    return result


@router.patch("/channels/{channel_id}")
async def toggle_channel(channel_id: str, body: ToggleChannelRequest):
    result = await run_in_threadpool(
        crawl_service.toggle_channel, channel_id, body.enabled
    )
    if result is None:
        raise HTTPException(404, f"Channel {channel_id} not found")
    return result


@router.post("/channels/{channel_id}/crawl")
async def crawl_channel(channel_id: str):
    try:
        return await run_in_threadpool(
            crawl_service.crawl_single_channel, channel_id
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        if "quota" in str(e).lower():
            raise HTTPException(429, str(e))
        raise HTTPException(500, str(e))


@router.post("/crawl-all")
async def crawl_all():
    try:
        return await run_in_threadpool(crawl_service.crawl_all_channels)
    except Exception as e:
        if "quota" in str(e).lower():
            raise HTTPException(429, str(e))
        raise HTTPException(500, str(e))


# ── Video endpoints ────────────────────────────────────────


@router.get("/videos")
async def list_videos(
    channel_id: str | None = Query(None),
    status: str | None = Query(None),
    layout_type: str | None = Query(None),
    order_by: str | None = Query(None),
    limit: int = Query(50, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    video_ids: str | None = Query(None),
):
    parsed_ids = video_ids.split(",") if video_ids else None
    return await run_in_threadpool(
        crawl_service.list_videos, channel_id, status, limit, offset, order_by, layout_type, parsed_ids
    )


@router.get("/videos/counts")
async def get_video_counts(channel_id: str | None = Query(None)):
    return await run_in_threadpool(crawl_service.get_video_counts, channel_id)


@router.patch("/videos/{video_id}/status")
async def update_video_status(video_id: str, body: UpdateStatusRequest):
    try:
        result = await run_in_threadpool(
            crawl_service.update_video_status, video_id, body.status, body.layout_type
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result is None:
        raise HTTPException(404, f"Video {video_id} not found")
    return result


@router.post("/videos/batch-status")
async def batch_update_status(body: BatchStatusRequest):
    try:
        count = await run_in_threadpool(
            crawl_service.batch_update_status, body.video_ids, body.status
        )
        return {"updated": count}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/videos/undo-auto-reject")
async def undo_auto_rejections(body: UndoAutoRejectRequest):
    count = await run_in_threadpool(
        crawl_service.undo_auto_rejections, body.video_ids
    )
    return {"restored": count}


# ── Frame inspection ─────────────────────────────────────


@router.post("/videos/{video_id}/inspect")
async def inspect_video(video_id: str):
    try:
        return await run_in_threadpool(inspect_service.inspect_single_video, video_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/videos/batch-inspect")
async def batch_inspect_videos(body: BatchInspectRequest):
    if not body.video_ids:
        raise HTTPException(400, "video_ids must not be empty")
    job_id = inspect_service.start_batch_job(body.video_ids)
    return {"job_id": job_id}


@router.get("/videos/inspect-job/{job_id}")
async def get_inspect_job(job_id: str):
    result = inspect_service.get_job_status(job_id)
    if result is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return result


# ── Screening ─────────────────────────────────────────────


@router.post("/screen")
async def screen_videos(body: ScreenRequest):
    try:
        return await run_in_threadpool(
            crawl_service.screen_videos, body.channel_id, body.limit
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Quota ──────────────────────────────────────────────────


@router.get("/quota")
async def get_quota():
    return await run_in_threadpool(crawl_service.get_quota_status)


# ── AI Classification ─────────────────────────────────────


@router.post("/auto-classify")
async def auto_classify(body: AutoClassifyRequest):
    try:
        return await run_in_threadpool(
            crawl_service.auto_classify_titles, body.channel_id, body.limit
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/classify")
async def classify_titles(body: ClassifyRequest):
    try:
        prompt = await run_in_threadpool(
            crawl_service.classify_titles, body.video_ids
        )
        return {"prompt": prompt}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/ai-screen")
async def ai_screen(body: AiScreenRequest):
    """Run lightweight AI screening on a batch of videos for the screening page."""
    import logging
    _logger = logging.getLogger(__name__)
    try:
        results = await run_in_threadpool(
            models_service.ai_screen_batch, body.video_ids, body.threshold
        )
        return {"results": results}
    except Exception as e:
        _logger.exception("ai-screen batch failed")
        raise HTTPException(500, f"AI screening failed: {type(e).__name__}: {e}")


# ── Annotations ──────────────────────────────────────────


# ── Single video ──────────────────────────────────────────


@router.get("/videos/{video_id}")
async def get_video(video_id: str):
    result = await run_in_threadpool(crawl_service.get_video, video_id)
    if result is None:
        raise HTTPException(404, f"Video {video_id} not found")
    return result


@router.get("/videos/{video_id}/download-status")
async def get_download_status(video_id: str):
    return await run_in_threadpool(crawl_service.get_download_status, video_id)


@router.post("/videos/{video_id}/download")
async def download_video(video_id: str):
    try:
        return await run_in_threadpool(crawl_service.download_single_video, video_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── Annotations ──────────────────────────────────────────


class SaveAnnotationsRequest(BaseModel):
    games: list[dict] | None = None
    notes: str | None = None


@router.get("/videos/{video_id}/annotations")
async def get_annotations(video_id: str):
    result = await run_in_threadpool(crawl_service.get_video_annotations, video_id)
    if result is None:
        return {"video_id": video_id, "annotations": None}
    return result


@router.put("/videos/{video_id}/annotations")
async def save_annotations(video_id: str, body: SaveAnnotationsRequest):
    try:
        result = await run_in_threadpool(
            crawl_service.save_video_annotations,
            video_id,
            {"games": body.games, "notes": body.notes},
        )
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))


# ── Video Clips ──────────────────────────────────────────


class CreateClipRequest(BaseModel):
    start_time: float = 0.0
    end_time: float | None = None
    label: str | None = None
    overlay_bbox: list[int]
    camera_bbox: list[int]
    ref_resolution: list[int] = [1920, 1080]
    board_flipped: bool = False
    board_theme: str = "lichess_default"


class UpdateClipRequest(BaseModel):
    start_time: float | None = None
    end_time: float | None = None
    label: str | None = None
    overlay_bbox: list[int] | None = None
    camera_bbox: list[int] | None = None
    ref_resolution: list[int] | None = None
    board_flipped: bool | None = None
    board_theme: str | None = None


@router.get("/videos/{video_id}/clips")
async def list_video_clips(video_id: str):
    return await run_in_threadpool(crawl_service.list_video_clips, video_id)


@router.post("/videos/{video_id}/clips")
async def create_video_clip(video_id: str, body: CreateClipRequest):
    try:
        return await run_in_threadpool(
            crawl_service.create_video_clip, video_id, body.model_dump()
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.put("/videos/{video_id}/clips/{clip_id}")
async def update_video_clip(video_id: str, clip_id: int, body: UpdateClipRequest):
    try:
        data = body.model_dump(exclude_unset=True)
        return await run_in_threadpool(
            crawl_service.update_video_clip, clip_id, data
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/videos/{video_id}/clips/{clip_id}")
async def delete_video_clip(video_id: str, clip_id: int):
    deleted = await run_in_threadpool(crawl_service.delete_video_clip, clip_id)
    if not deleted:
        raise HTTPException(404, f"Clip {clip_id} not found")
    return {"deleted": True}
