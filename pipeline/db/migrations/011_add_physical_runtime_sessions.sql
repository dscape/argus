-- Physical runtime evaluation sessions
CREATE TABLE IF NOT EXISTS physical_runtime_sessions (
    id                 TEXT PRIMARY KEY,
    created_at         TIMESTAMPTZ DEFAULT now(),
    sample_size        INTEGER NOT NULL,
    square_accuracy    FLOAT,
    non_empty_accuracy FLOAT,
    exact_match_rate   FLOAT,
    results            JSONB NOT NULL,
    pin_state          JSONB DEFAULT '{}',
    evaluation_id      INTEGER REFERENCES model_evaluations(id)
);

CREATE INDEX IF NOT EXISTS idx_physical_runtime_sessions_created
    ON physical_runtime_sessions (created_at DESC);
