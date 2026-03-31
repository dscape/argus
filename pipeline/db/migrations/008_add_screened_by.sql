-- Track who made the screening decision: 'human' or 'ai'
ALTER TABLE youtube_videos
    ADD COLUMN IF NOT EXISTS screened_by VARCHAR(10);

-- Backfill: AI auto-decided videos
UPDATE youtube_videos
SET screened_by = 'ai'
WHERE ai_screening_auto_decided = true
  AND screening_status IS NOT NULL
  AND screened_by IS NULL;

-- Backfill: all other labeled videos were human-screened
UPDATE youtube_videos
SET screened_by = 'human'
WHERE screening_status IS NOT NULL
  AND screened_by IS NULL;
