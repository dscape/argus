"""Endpoints for browsing physical board failure-study bundles."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api.services.evaluate import physical_failure_study_service

router = APIRouter()


class UpdateFailureStudyEntryRequest(BaseModel):
    study_path: str
    selected_index: int
    final_bucket: str | None = None
    notes: str | None = None


@router.get("/studies")
async def list_failure_studies() -> dict[str, Any]:
    studies = await run_in_threadpool(physical_failure_study_service.list_failure_studies)
    return {"studies": studies}


@router.get("/study")
async def get_failure_study(path: str) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            physical_failure_study_service.get_failure_study,
            path,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/context")
async def get_failure_study_context(
    path: str,
    selected_index: int,
    context_frames: int = 10,
    image_max_side: int = 720,
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            physical_failure_study_service.get_failure_study_context,
            path,
            selected_index=selected_index,
            context_frames=context_frames,
            image_max_side=image_max_side,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.patch("/entry")
async def update_failure_study_entry(
    body: UpdateFailureStudyEntryRequest,
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            physical_failure_study_service.update_failure_study_entry,
            body.study_path,
            selected_index=body.selected_index,
            final_bucket=body.final_bucket,
            notes=body.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
