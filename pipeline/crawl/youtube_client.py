"""YouTube Data API v3 client with exponential backoff."""

import os
import random
import time
import logging

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from pipeline.crawl.quota_tracker import QuotaTracker

logger = logging.getLogger(__name__)


class YouTubeClient:
    """Wrapper around YouTube Data API v3 with retry logic and quota tracking."""

    def __init__(self, quota_tracker: QuotaTracker | None = None):
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY not set in environment")
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.quota = quota_tracker or QuotaTracker()
        self.max_retries = 5
        self.initial_backoff = 1.0
        self.max_backoff = 60.0

    def _execute_with_retry(self, request, quota_cost: int, endpoint: str):
        """Execute an API request with exponential backoff on 403/429."""
        for attempt in range(self.max_retries):
            try:
                self.quota.check_or_halt(quota_cost)
                response = request.execute()
                self.quota.log_call(
                    api_name="youtube",
                    endpoint=endpoint,
                    quota_cost=quota_cost,
                )
                return response
            except HttpError as e:
                if e.resp.status in (403, 429):
                    if attempt == self.max_retries - 1:
                        raise
                    backoff = min(
                        self.initial_backoff * (2 ** attempt),
                        self.max_backoff,
                    )
                    jitter = random.uniform(0, backoff * 0.5)
                    wait = backoff + jitter
                    logger.warning(
                        f"HTTP {e.resp.status} on {endpoint}, "
                        f"retrying in {wait:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                else:
                    raise

    def get_channel_by_handle(self, handle: str) -> dict | None:
        """Resolve a @handle to channel info. Cost: 1 unit."""
        clean_handle = handle.lstrip("@")
        request = self.youtube.channels().list(
            part="contentDetails,snippet",
            forHandle=clean_handle,
        )
        response = self._execute_with_retry(request, 1, "channels.list(forHandle)")
        items = response.get("items", [])
        if not items:
            return None
        item = items[0]
        return {
            "channel_id": item["id"],
            "title": item["snippet"]["title"],
            "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
        }

    def get_channel_by_id(self, channel_id: str) -> dict | None:
        """Get channel info by UC-prefixed ID. Cost: 1 unit."""
        request = self.youtube.channels().list(
            part="contentDetails,snippet",
            id=channel_id,
        )
        response = self._execute_with_retry(request, 1, "channels.list(id)")
        items = response.get("items", [])
        if not items:
            return None
        item = items[0]
        return {
            "channel_id": item["id"],
            "title": item["snippet"]["title"],
            "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
        }

    def search_channels(self, query: str, max_results: int = 5) -> list[dict]:
        """Search for channels by name. Cost: 100 units. Use sparingly."""
        request = self.youtube.search().list(
            part="snippet",
            q=query,
            type="channel",
            maxResults=max_results,
        )
        response = self._execute_with_retry(request, 100, "search.list(channel)")
        results = []
        for item in response.get("items", []):
            results.append({
                "channel_id": item["snippet"]["channelId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
            })
        return results

    def list_playlist_items(
        self,
        playlist_id: str,
        page_token: str | None = None,
        max_results: int = 50,
    ) -> dict:
        """Fetch a page of videos from a playlist. Cost: 1 unit per page."""
        kwargs = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": max_results,
        }
        if page_token:
            kwargs["pageToken"] = page_token

        request = self.youtube.playlistItems().list(**kwargs)
        return self._execute_with_retry(request, 1, "playlistItems.list")

    def get_video_details(self, video_ids: list[str]) -> list[dict]:
        """Fetch video details for up to 50 video IDs. Cost: 1 unit."""
        request = self.youtube.videos().list(
            part="snippet,contentDetails",
            id=",".join(video_ids[:50]),
        )
        response = self._execute_with_retry(request, 1, "videos.list")
        return response.get("items", [])
