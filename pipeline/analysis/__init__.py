"""Shared local-video analysis helpers."""

from pipeline.analysis.config import VideoAnalysisConfig

__all__ = ["VideoAnalysisConfig", "VideoAnalysisPipeline"]


def __getattr__(name: str):
    if name == "VideoAnalysisPipeline":
        from pipeline.analysis.pipeline import VideoAnalysisPipeline

        return VideoAnalysisPipeline
    raise AttributeError(name)
