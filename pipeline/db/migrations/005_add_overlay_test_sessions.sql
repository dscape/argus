-- Overlay test sessions (piece classifier accuracy testing)
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
