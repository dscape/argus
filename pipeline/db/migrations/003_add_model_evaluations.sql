-- Track model evaluation runs over time for regression detection.

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
