-- Segmentation evaluation sessions (shareable inspector runs)
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
