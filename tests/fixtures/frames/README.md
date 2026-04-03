# Overlay Detection Test Fixtures

Ground truth frames for testing `fast_overlay_check()` and grid detection.

## Resolution Requirement

All frames MUST be 1920x1080. This is the native YouTube video resolution.

YouTube auto-generated thumbnails (`maxresdefault.jpg`) are only 1280x720.
The resolution difference affects grid detection thresholds (cell variance,
Sobel peak spacing, checkerboard validation). Do NOT use YouTube thumbnails
as test fixtures — extract frames from actual video using yt-dlp instead.

## Structure

```
tests/fixtures/frames/
  {video_id}/
    25pct.jpg    # Frame at ~25% of video duration
    50pct.jpg    # Frame at ~50% of video duration
    75pct.jpg    # Frame at ~75% of video duration
  ground_truth.json
```

## Ground Truth Format

`ground_truth.json` maps `"{video_id}/{label}"` to:

| Field          | Type              | Description                          |
|----------------|-------------------|--------------------------------------|
| `image`        | `str`             | Relative path to frame file          |
| `has_overlay`  | `bool`            | Whether a chess overlay is present   |
| `bbox`         | `[x,y,w,h]|null` | Overlay bounding box (null if none)  |
| `frame_width`  | `int`             | Always 1920                          |
| `frame_height` | `int`             | Always 1080                          |
| `annotated_at` | `str`             | ISO timestamp of annotation          |

## How to Add Fixtures

1. Use `python -m pipeline fetch-frames --resolution fullres` to download
   frames at native 1920x1080 via yt-dlp
2. Copy frames to `tests/fixtures/frames/{video_id}/`
3. Annotate bounding boxes via the dev-tools UI at `/annotate/overlay-bbox`
4. Copy the annotation from `data/overlay/dataset/frames/ground_truth.json`
   into `tests/fixtures/frames/ground_truth.json`
