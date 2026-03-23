-- Migration 001: Add video_clips table and clip_id to training_clips
-- Run against an existing database to add clip segmentation support.

CREATE TABLE IF NOT EXISTS video_clips (
    id              SERIAL PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES youtube_videos(video_id),
    clip_index      INTEGER NOT NULL,
    label           TEXT,
    start_time      FLOAT NOT NULL DEFAULT 0.0,
    end_time        FLOAT,
    overlay_bbox    JSONB NOT NULL,
    camera_bbox     JSONB NOT NULL,
    ref_resolution  JSONB NOT NULL,
    board_flipped   BOOLEAN NOT NULL DEFAULT false,
    board_theme     VARCHAR(30) NOT NULL DEFAULT 'lichess_default',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (video_id, clip_index)
);

CREATE INDEX IF NOT EXISTS idx_video_clips_video
    ON video_clips (video_id);

-- Add clip_id FK to training_clips (nullable for backward compat)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'training_clips' AND column_name = 'clip_id'
    ) THEN
        ALTER TABLE training_clips ADD COLUMN clip_id INTEGER REFERENCES video_clips(id);
    END IF;
END $$;
