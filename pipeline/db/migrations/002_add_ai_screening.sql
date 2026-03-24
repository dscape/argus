-- Add AI screening metadata columns to youtube_videos.
-- These track the automated classifier's predictions alongside human decisions.

ALTER TABLE youtube_videos
    ADD COLUMN IF NOT EXISTS ai_screening_class VARCHAR(20),
    ADD COLUMN IF NOT EXISTS ai_screening_confidence FLOAT,
    ADD COLUMN IF NOT EXISTS ai_screening_auto_decided BOOLEAN DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_youtube_videos_ai_screening
    ON youtube_videos (ai_screening_class);
