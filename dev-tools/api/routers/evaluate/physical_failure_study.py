"""Endpoints for browsing physical board failure-study bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from api.services.evaluate import physical_failure_study_service

router = APIRouter()


class UpdateFailureStudyEntryRequest(BaseModel):
    study_path: str
    episode_id: str
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


@router.patch("/entry")
async def update_failure_study_entry(
    body: UpdateFailureStudyEntryRequest,
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            physical_failure_study_service.update_failure_study_entry,
            body.study_path,
            episode_id=body.episode_id,
            final_bucket=body.final_bucket,
            notes=body.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/image")
async def get_failure_study_image(path: str, image: str) -> FileResponse:
    try:
        resolved = await run_in_threadpool(
            physical_failure_study_service.resolve_image_path,
            path,
            image,
        )
        return FileResponse(resolved)
    except PermissionError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/export-csv")
async def export_failure_study_csv(path: str) -> Response:
    try:
        content = await run_in_threadpool(
            physical_failure_study_service.export_manual_buckets_csv,
            path,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    filename = f"{Path(path).name}_manual_buckets.csv"
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
