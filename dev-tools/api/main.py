"""Argus Dev Tools — FastAPI backend.

Wraps existing pipeline.overlay.* modules to provide a REST API
for the developer tools web UI.
"""

import logging

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
from fastapi.middleware.cors import CORSMiddleware
from pipeline.db.connection import migrate

from api.routers.annotate import (
    calibration,
    overlay_bbox,
    physical_eval,
    physical_train,
    video_session,
)
from api.routers.data import clips, real, synthetic
from api.routers.evaluate import models, overlay
from api.routers.videos import crawl

logger = logging.getLogger(__name__)

app = FastAPI(title="Argus API", version="0.1.0", redirect_slashes=False)


@app.on_event("startup")
def run_migrations():
    try:
        migrate()
    except Exception as e:
        logger.warning("Migration failed (database may be unavailable): %s", e)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)

app.include_router(overlay.router, prefix="/api/overlay", tags=["overlay"])
app.include_router(calibration.router, prefix="/api/calibration", tags=["calibration"])
app.include_router(overlay_bbox.router, prefix="/api/overlay-bbox", tags=["overlay-bbox"])
app.include_router(physical_eval.router, prefix="/api/physical-eval", tags=["physical-eval"])
app.include_router(physical_train.router, prefix="/api/physical-train", tags=["physical-train"])
app.include_router(clips.router, prefix="/api/clips", tags=["clips"])
app.include_router(video_session.router, prefix="/api/video", tags=["video"])
app.include_router(synthetic.router, prefix="/api/synthetic", tags=["synthetic"])
app.include_router(real.router, prefix="/api/real-data", tags=["real-data"])
app.include_router(crawl.router, prefix="/api/crawl", tags=["crawl"])
app.include_router(models.router, prefix="/api/models", tags=["models"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
