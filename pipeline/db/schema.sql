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
    -- NULL = unscreened, 'approved' = has overlay + OTB, 'rejected' = failed screening
    screening_confidence FLOAT,
    overlay_bbox        JSONB,
    -- {"x": ..., "y": ..., "w": ..., "h": ...}
    has_otb_footage     BOOLEAN,
    layout_type         VARCHAR(20),
    -- 'overlay' (has 2D board overlay), 'otb_only' (camera only)
    annotations         JSONB,
    -- {"games": [...], "notes": "..."}

    -- AI screening classifier predictions (set by ai_screen_batch)
    ai_screening_class       VARCHAR(20),
    ai_screening_confidence  FLOAT,
    ai_screening_auto_decided BOOLEAN DEFAULT false,

    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_youtube_videos_channel
    ON youtube_videos (channel_id);
CREATE INDEX IF NOT EXISTS idx_youtube_videos_published
    ON youtube_videos (published_at);
CREATE INDEX IF NOT EXISTS idx_youtube_videos_screening
    ON youtube_videos (screening_status);
CREATE INDEX IF NOT EXISTS idx_youtube_videos_ai_screening
    ON youtube_videos (ai_screening_class);

-- ============================================================
-- Video clips (manual time segments with per-clip calibration)
-- ============================================================
CREATE TABLE IF NOT EXISTS video_clips (
    id                  SERIAL PRIMARY KEY,
    video_id            TEXT NOT NULL REFERENCES youtube_videos(video_id),
    clip_index          INTEGER NOT NULL,
    label               TEXT,
    start_time          FLOAT NOT NULL DEFAULT 0.0,
    end_time            FLOAT,
    overlay_bbox        JSONB NOT NULL,
    camera_bbox         JSONB NOT NULL,
    ref_resolution      JSONB NOT NULL,
    board_flipped       BOOLEAN NOT NULL DEFAULT false,
    board_theme         VARCHAR(30) NOT NULL DEFAULT 'lichess_default',
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (video_id, clip_index)
);

CREATE INDEX IF NOT EXISTS idx_video_clips_video
    ON video_clips (video_id);

-- ============================================================
-- Training clips (generated from overlay pipeline)
-- ============================================================
CREATE TABLE IF NOT EXISTS training_clips (
    id                  SERIAL PRIMARY KEY,
    video_id            TEXT NOT NULL REFERENCES youtube_videos(video_id),
    clip_id             INTEGER REFERENCES video_clips(id),
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

-- ============================================================
-- Model evaluation history (regression tracking)
-- ============================================================
CREATE TABLE IF NOT EXISTS model_evaluations (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,
    evaluated_at    TIMESTAMPTZ DEFAULT now(),
    sample_size     INTEGER NOT NULL,
    accuracy        FLOAT,
    precision_avg   FLOAT,
    recall_avg      FLOAT,
    f1_avg          FLOAT,
    per_class       JSONB,
    threshold       FLOAT,
    auto_rate       FLOAT,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_evaluations_model
    ON model_evaluations (model_name, evaluated_at);

-- ============================================================
-- Screening sessions (shareable inspector evaluation runs)
-- ============================================================
CREATE TABLE IF NOT EXISTS screening_sessions (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now(),
    sample_size     INTEGER NOT NULL,
    model_version   TEXT,
    accuracy        FLOAT,
    per_class       JSONB,
    results         JSONB NOT NULL,
    pin_state       JSONB DEFAULT '{}',
    evaluation_id   INTEGER REFERENCES model_evaluations(id)
);

CREATE INDEX IF NOT EXISTS idx_screening_sessions_created
    ON screening_sessions (created_at DESC);

-- ============================================================
-- Overlay test sessions (piece classifier accuracy testing)
-- ============================================================
CREATE TABLE IF NOT EXISTS overlay_test_sessions (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now(),
    sample_size     INTEGER NOT NULL,
    accuracy        FLOAT,
    piece_accuracy  FLOAT,
    results         JSONB NOT NULL,
    pin_state       JSONB DEFAULT '{}',
    evaluation_id   INTEGER REFERENCES model_evaluations(id)
);

CREATE INDEX IF NOT EXISTS idx_overlay_test_sessions_created
    ON overlay_test_sessions (created_at DESC);

-- ============================================================
-- Segmentation evaluation sessions
-- ============================================================
CREATE TABLE IF NOT EXISTS segmentation_eval_sessions (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now(),
    sample_size     INTEGER NOT NULL,
    segment_consistency   FLOAT,
    gap_consistency       FLOAT,
    piece_readability     FLOAT,
    false_negative_rate   FLOAT,
    coverage_ratio        FLOAT,
    results         JSONB NOT NULL,
    pin_state       JSONB DEFAULT '{}',
    evaluation_id   INTEGER REFERENCES model_evaluations(id)
);

CREATE INDEX IF NOT EXISTS idx_segmentation_eval_sessions_created
    ON segmentation_eval_sessions (created_at DESC);

-- ============================================================
-- Calibration evaluation sessions
-- ============================================================
CREATE TABLE IF NOT EXISTS calibration_eval_sessions (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now(),
    sample_size     INTEGER NOT NULL,
    overlay_iou_avg       FLOAT,
    theme_accuracy        FLOAT,
    orientation_accuracy  FLOAT,
    grid_success_rate     FLOAT,
    fen_validity_rate     FLOAT,
    results         JSONB NOT NULL,
    pin_state       JSONB DEFAULT '{}',
    evaluation_id   INTEGER REFERENCES model_evaluations(id)
);

CREATE INDEX IF NOT EXISTS idx_calibration_eval_sessions_created
    ON calibration_eval_sessions (created_at DESC);
