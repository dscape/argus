CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- FIDE players (from players.zip)
-- ============================================================
CREATE TABLE IF NOT EXISTS fide_players (
    fide_id         INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    federation      VARCHAR(3),
    title           VARCHAR(4),
    standard_rating INTEGER,
    rapid_rating    INTEGER,
    blitz_rating    INTEGER,
    birth_year      INTEGER,
    slug            TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fide_players_name_trgm
    ON fide_players USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fide_players_name_lower
    ON fide_players (lower(name));

-- ============================================================
-- Player aliases for fuzzy matching
-- ============================================================
CREATE TABLE IF NOT EXISTS player_aliases (
    id              SERIAL PRIMARY KEY,
    alias           TEXT NOT NULL,
    fide_id         INTEGER NOT NULL REFERENCES fide_players(fide_id),
    source          VARCHAR(20) DEFAULT 'import',
    UNIQUE(alias, fide_id)
);

CREATE INDEX IF NOT EXISTS idx_player_aliases_trgm
    ON player_aliases USING gin (alias gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_player_aliases_fide_id
    ON player_aliases (fide_id);

-- ============================================================
-- Games (from pgns.zip -- ~3.3M rows)
-- ============================================================
CREATE TABLE IF NOT EXISTS games (
    id              SERIAL PRIMARY KEY,
    white_name      TEXT,
    black_name      TEXT,
    white_fide_id   INTEGER,
    black_fide_id   INTEGER,
    event           TEXT,
    site            TEXT,
    date            DATE,
    round           TEXT,
    result          VARCHAR(7),
    white_elo       INTEGER,
    black_elo       INTEGER,
    eco             VARCHAR(5),
    pgn_moves       TEXT NOT NULL,
    twic_file       TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_games_white_fide_id ON games (white_fide_id);
CREATE INDEX IF NOT EXISTS idx_games_black_fide_id ON games (black_fide_id);
CREATE INDEX IF NOT EXISTS idx_games_date ON games (date);
CREATE INDEX IF NOT EXISTS idx_games_event_trgm
    ON games USING gin (event gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_games_players
    ON games (white_fide_id, black_fide_id, date);

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

    extracted_white     TEXT,
    extracted_black     TEXT,
    white_fide_id       INTEGER,
    black_fide_id       INTEGER,
    extracted_event     TEXT,
    extracted_year      INTEGER,
    extracted_round     TEXT,
    extracted_result    VARCHAR(7),
    extraction_method   VARCHAR(20),
    extraction_confidence FLOAT,

    has_pgn             BOOLEAN DEFAULT false,
    video_pgn           TEXT,

    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_youtube_videos_channel
    ON youtube_videos (channel_id);
CREATE INDEX IF NOT EXISTS idx_youtube_videos_published
    ON youtube_videos (published_at);
CREATE INDEX IF NOT EXISTS idx_youtube_videos_players
    ON youtube_videos (white_fide_id, black_fide_id, extracted_year);
CREATE INDEX IF NOT EXISTS idx_youtube_videos_event_trgm
    ON youtube_videos USING gin (extracted_event gin_trgm_ops);

-- ============================================================
-- Video chapters (for multi-board streams)
-- ============================================================
CREATE TABLE IF NOT EXISTS video_chapters (
    id                  SERIAL PRIMARY KEY,
    video_id            TEXT NOT NULL REFERENCES youtube_videos(video_id),
    timestamp_seconds   INTEGER NOT NULL,
    chapter_title       TEXT NOT NULL,

    extracted_white     TEXT,
    extracted_black     TEXT,
    white_fide_id       INTEGER,
    black_fide_id       INTEGER,
    extraction_confidence FLOAT,

    UNIQUE(video_id, timestamp_seconds)
);

CREATE INDEX IF NOT EXISTS idx_video_chapters_video
    ON video_chapters (video_id);
CREATE INDEX IF NOT EXISTS idx_video_chapters_players
    ON video_chapters (white_fide_id, black_fide_id);

-- ============================================================
-- Game-video links (matching output)
-- ============================================================
CREATE TABLE IF NOT EXISTS game_video_links (
    id                  SERIAL PRIMARY KEY,
    game_id             INTEGER NOT NULL REFERENCES games(id),
    video_id            TEXT NOT NULL REFERENCES youtube_videos(video_id),
    chapter_id          INTEGER REFERENCES video_chapters(id),
    timestamp_seconds   INTEGER,
    match_confidence    FLOAT NOT NULL,
    match_signals       JSONB NOT NULL,
    verified            BOOLEAN DEFAULT false,
    created_at          TIMESTAMPTZ DEFAULT now(),

    UNIQUE(game_id, video_id)
);

CREATE INDEX IF NOT EXISTS idx_game_video_links_video
    ON game_video_links (video_id);
CREATE INDEX IF NOT EXISTS idx_game_video_links_confidence
    ON game_video_links (match_confidence DESC);

-- ============================================================
-- Training clips (Phase 6 output)
-- ============================================================
CREATE TABLE IF NOT EXISTS training_clips (
    id                  SERIAL PRIMARY KEY,
    game_id             INTEGER NOT NULL REFERENCES games(id),
    video_id            TEXT NOT NULL REFERENCES youtube_videos(video_id),
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
-- Overlay pipeline: layout type tag for video screening
-- ============================================================
ALTER TABLE youtube_videos ADD COLUMN IF NOT EXISTS layout_type VARCHAR(20);
-- Values: NULL (unscreened), 'overlay' (has 2D board overlay), 'otb_only' (camera only)
