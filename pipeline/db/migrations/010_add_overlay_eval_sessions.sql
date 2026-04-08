-- Overlay detection evaluation sessions
CREATE TABLE IF NOT EXISTS overlay_eval_sessions (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now(),
    sample_size     INTEGER NOT NULL,
    detection_rate  FLOAT,
    fen_success_rate FLOAT,
    results         JSONB NOT NULL,
    pin_state       JSONB DEFAULT '{}',
    evaluation_id   INTEGER REFERENCES model_evaluations(id)
);

CREATE INDEX IF NOT EXISTS idx_overlay_eval_sessions_created
    ON overlay_eval_sessions (created_at DESC);
