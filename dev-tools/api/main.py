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

from api.routers import calibration, clips, crawl, models, overlay, synthetic, video

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
app.include_router(clips.router, prefix="/api/clips", tags=["clips"])
app.include_router(video.router, prefix="/api/video", tags=["video"])
app.include_router(synthetic.router, prefix="/api/synthetic", tags=["synthetic"])
app.include_router(crawl.router, prefix="/api/crawl", tags=["crawl"])
app.include_router(models.router, prefix="/api/models", tags=["models"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
