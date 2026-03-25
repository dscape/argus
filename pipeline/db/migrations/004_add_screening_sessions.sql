-- Screening sessions: persist inspector evaluation runs for sharing
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
