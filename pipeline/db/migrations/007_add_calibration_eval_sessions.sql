-- Calibration evaluation sessions (shareable inspector runs)
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
