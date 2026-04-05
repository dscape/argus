# Overlay Detection Test Fixtures

Ground truth frames for testing `fast_overlay_check()` and grid detection.

## Resolution Requirement

Fixtures should come from extracted video frames, not YouTube thumbnails.
The set now includes a mix of native `1920x1080` frames and extracted
`1280x720` frames, so each entry's `frame_width` / `frame_height` in
`ground_truth.json` is the source of truth.

When adding fixtures, copy the frame tier whose actual image size matches
the annotation stored in the ground-truth file.

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
| `frame_width`  | `int`             | Actual frame width for that image    |
| `frame_height` | `int`             | Actual frame height for that image   |
| `annotated_at` | `str`             | ISO timestamp of annotation          |

## How to Add Fixtures

1. Use `python -m pipeline fetch-frames` to download extracted video frames
2. Copy frames to `tests/fixtures/frames/{video_id}/`
3. Annotate bounding boxes via the dev-tools UI at `/annotate/overlay-bbox`
4. Copy the annotation from `data/videos/ground_truth.json`
   into `tests/fixtures/frames/ground_truth.json`
