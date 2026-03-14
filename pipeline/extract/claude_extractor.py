"""Use Claude API to extract metadata from non-standard video titles."""

import json
import logging
import os

import anthropic

from pipeline.db.connection import get_conn

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract chess game information from this YouTube video title and description snippet.
Return a JSON object with these fields (use null for unknown):
- white: white player name (string or null)
- black: black player name (string or null)
- event: tournament/event name (string or null)
- year: year as integer (or null)
- round: round number as string (or null)
- result: game result as "1-0", "0-1", or "1/2-1/2" (or null)
- confidence: your confidence 0.0-1.0

Only return the JSON, no other text.

Title: {title}
Description (first 500 chars): {description}"""


class ClaudeExtractor:
    """Extract metadata from video titles using Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        batch_size: int = 50,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    def extract_batch(
        self,
        videos: list[dict],
    ) -> list[dict]:
        """Extract metadata from a batch of videos.

        Args:
            videos: List of dicts with 'video_id', 'title', 'description'.

        Returns:
            List of dicts with extracted fields + 'video_id'.
        """
        results = []

        for video in videos:
            title = video["title"]
            desc = (video.get("description") or "")[:500]

            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    messages=[{
                        "role": "user",
                        "content": EXTRACTION_PROMPT.format(
                            title=title,
                            description=desc,
                        ),
                    }],
                )

                response_text = message.content[0].text.strip()

                # Parse JSON from response
                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("\n", 1)[1]
                    response_text = response_text.rsplit("```", 1)[0]

                data = json.loads(response_text)
                data["video_id"] = video["video_id"]
                results.append(data)

            except (json.JSONDecodeError, anthropic.APIError) as e:
                logger.warning(f"Claude extraction failed for {video['video_id']}: {e}")
                results.append({
                    "video_id": video["video_id"],
                    "white": None,
                    "black": None,
                    "event": None,
                    "year": None,
                    "round": None,
                    "result": None,
                    "confidence": 0.0,
                })

        return results

    def extract_and_store(self, min_existing_confidence: float = 0.5):
        """Fetch low-confidence videos from DB, extract with Claude, and update."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT video_id, title, description
                    FROM youtube_videos
                    WHERE (extraction_confidence IS NULL
                           OR extraction_confidence < %s)
                      AND extraction_method IS DISTINCT FROM 'claude'
                    ORDER BY published_at DESC
                    LIMIT %s
                    """,
                    (min_existing_confidence, self.batch_size),
                )
                videos = [
                    {"video_id": r[0], "title": r[1], "description": r[2]}
                    for r in cur.fetchall()
                ]

        if not videos:
            print("No videos need Claude extraction.")
            return

        print(f"Extracting metadata for {len(videos)} videos with Claude...")
        results = self.extract_batch(videos)

        updated = 0
        with get_conn() as conn:
            with conn.cursor() as cur:
                for data in results:
                    if data.get("confidence", 0) > 0:
                        cur.execute(
                            """
                            UPDATE youtube_videos
                            SET extracted_white = %s,
                                extracted_black = %s,
                                extracted_event = %s,
                                extracted_year = %s,
                                extracted_round = %s,
                                extracted_result = %s,
                                extraction_method = 'claude',
                                extraction_confidence = %s,
                                updated_at = now()
                            WHERE video_id = %s
                            """,
                            (
                                data.get("white"),
                                data.get("black"),
                                data.get("event"),
                                data.get("year"),
                                data.get("round"),
                                data.get("result"),
                                data.get("confidence", 0.0),
                                data["video_id"],
                            ),
                        )
                        updated += 1
                conn.commit()

        print(f"Updated {updated}/{len(videos)} videos with Claude extraction.")
