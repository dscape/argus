"""Crawl management endpoints — channels, videos, title scoring, AI classification."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.services import crawl_service

router = APIRouter()


# ── Request models ──────────────────────────────────────────


class AddChannelRequest(BaseModel):
    handle: str


class ToggleChannelRequest(BaseModel):
    enabled: bool


class UpdateStatusRequest(BaseModel):
    status: str | None = None


class BatchStatusRequest(BaseModel):
    video_ids: list[str]
    status: str


class ClassifyRequest(BaseModel):
    video_ids: list[str]


# ── Channel endpoints ──────────────────────────────────────


@router.get("/channels")
async def list_channels():
    return await run_in_threadpool(crawl_service.list_channels)


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
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    return await run_in_threadpool(
        crawl_service.list_videos, channel_id, status, limit, offset
    )


@router.patch("/videos/{video_id}/status")
async def update_video_status(video_id: str, body: UpdateStatusRequest):
    try:
        result = await run_in_threadpool(
            crawl_service.update_video_status, video_id, body.status
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


# ── Quota ──────────────────────────────────────────────────


@router.get("/quota")
async def get_quota():
    return await run_in_threadpool(crawl_service.get_quota_status)


# ── AI Classification ─────────────────────────────────────


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
