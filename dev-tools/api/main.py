"""Argus Dev Tools — FastAPI backend.

Wraps existing pipeline.overlay.* modules to provide a REST API
for the developer tools web UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import overlay, calibration, clips, video, synthetic

app = FastAPI(title="Argus Dev Tools API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(overlay.router, prefix="/api/overlay", tags=["overlay"])
app.include_router(calibration.router, prefix="/api/calibration", tags=["calibration"])
app.include_router(clips.router, prefix="/api/clips", tags=["clips"])
app.include_router(video.router, prefix="/api/video", tags=["video"])
app.include_router(synthetic.router, prefix="/api/synthetic", tags=["synthetic"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
