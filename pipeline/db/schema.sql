CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- YouTube crawl channels (config table, not hardcoded)
-- ============================================================
CREATE TABLE IF NOT EXISTS crawl_channels (
    channel_id          TEXT PRIMARY KEY,
    channel_handle      TEXT,
    channel_name        TEXT NOT NULL,
    tier                INTEGER NOT NULL DEFAULT 3,
    uploads_playlist_id TEXT,
    last_crawled_at     TIMESTAMPTZ,
    enabled             BOOLEAN DEFAULT true,
    notes               TEXT
);

-- ============================================================
-- Raw YouTube API responses (re-parseable)
-- ============================================================
CREATE TABLE IF NOT EXISTS youtube_api_raw (
    id              SERIAL PRIMARY KEY,
    channel_id      TEXT NOT NULL,
    playlist_id     TEXT NOT NULL,
    page_token      TEXT,
    response_json   JSONB NOT NULL,
    fetched_at      TIMESTAMPTZ DEFAULT now(),
    quota_cost      INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_youtube_api_raw_channel
    ON youtube_api_raw (channel_id);

-- ============================================================
-- YouTube videos (parsed from raw responses)
-- ============================================================
CREATE TABLE IF NOT EXISTS youtube_videos (
    video_id            TEXT PRIMARY KEY,
    channel_id          TEXT NOT NULL,
    channel_handle      TEXT,
    title               TEXT NOT NULL,
    description         TEXT,
    published_at        TIMESTAMPTZ,
    tags                TEXT[],

    -- Screening results (set by pipeline screen stage)
    screening_status    VARCHAR(20),
    -- NULL = unscreened, 'candidate' = title matched,
    -- 'approved' = has overlay + OTB, 'rejected' = failed screening
    screening_confidence FLOAT,
    overlay_bbox        JSONB,
    -- {"x": ..., "y": ..., "w": ..., "h": ...}
    has_otb_footage     BOOLEAN,
    layout_type         VARCHAR(20),
    -- 'overlay' (has 2D board overlay), 'otb_only' (camera only)

    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_youtube_videos_channel
    ON youtube_videos (channel_id);
CREATE INDEX IF NOT EXISTS idx_youtube_videos_published
    ON youtube_videos (published_at);
CREATE INDEX IF NOT EXISTS idx_youtube_videos_screening
    ON youtube_videos (screening_status);

-- ============================================================
-- Training clips (generated from overlay pipeline)
-- ============================================================
CREATE TABLE IF NOT EXISTS training_clips (
    id                  SERIAL PRIMARY KEY,
    video_id            TEXT NOT NULL REFERENCES youtube_videos(video_id),
    game_index          INTEGER NOT NULL DEFAULT 0,
    file_path           TEXT NOT NULL,
    num_frames          INTEGER NOT NULL,
    num_moves           INTEGER NOT NULL,
    alignment_quality   FLOAT,
    has_errors          BOOLEAN DEFAULT false,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- API quota tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS api_quota_log (
    id              SERIAL PRIMARY KEY,
    api_name        VARCHAR(30) NOT NULL,
    endpoint        TEXT NOT NULL,
    quota_cost      INTEGER NOT NULL,
    logged_at       TIMESTAMPTZ DEFAULT now(),
    details         JSONB
);

CREATE INDEX IF NOT EXISTS idx_api_quota_log_date
    ON api_quota_log (logged_at);
