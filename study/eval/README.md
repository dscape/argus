# Eval set schema

Final eval layout:

- `study/eval/frames/`
- `study/eval/labels.jsonl`

## Final labels schema

Each `labels.jsonl` row looks like:

```json
{
  "frame_id": "7RaBQag34Hk_016347",
  "image_path": "study/eval/frames/7RaBQag34Hk_016347.jpg",
  "category": "mid-move",
  "corners": [[255.0, 90.0], [415.0, 162.0], [149.0, 224.0], [51.0, 113.0]],
  "pieces": [
    {"type": "K", "square": "g1"},
    {"type": "q", "square": "d8"},
    {"type": "N", "square": null}
  ],
  "source_video_id": "7RaBQag34Hk",
  "source_frame_index": 16347,
  "notes": "white knight hovering during capture"
}
```

Rules:

- `category` must be one of:
  - `a-file-rook`
  - `lateral-occlusion`
  - `low-camera-angle`
  - `dense-middlegame`
  - `mid-move`
  - `easy-stationary`
- `corners` are full-frame board corners in image pixels, ordered TL/TR/BR/BL.
- `pieces[].type` uses Argus piece symbols: `P N B R Q K p n b r q k`.
- `pieces[].square` is algebraic (`a1`..`h8`) for placed pieces.
- `pieces[].square = null` means the piece is visible but not rooted on any square.

## Bootstrap workflow

Build a candidate manifest from existing held-out physical annotations:

```bash
./.venv/bin/python study/eval/build_candidates.py
```

That writes `study/eval/candidates.jsonl` with starter piece labels and source-frame metadata.

Each candidate row looks like:

```json
{
  "frame_id": "clip_overlay_..._frame0012",
  "annotation_id": "clip_overlay_..._frame0012",
  "clip_path": "data/argus/train_real/clip_overlay_....pt",
  "frame_index": 12,
  "source_video_id": "7RaBQag34Hk",
  "source_frame_index": 16482,
  "corners": [[255.0, 90.0], [415.0, 162.0], [149.0, 224.0], [51.0, 113.0]],
  "pieces": [
    {"type": "q", "square": "c8"},
    {"type": "k", "square": "a1"}
  ],
  "materialized_image_path": "study/eval/frames/clip_overlay_..._frame0012.jpg"
}
```

Curate a `study/eval/selection.jsonl` file from those candidates.

Each selection row looks like:

```json
{
  "frame_id": "clip_overlay_..._frame0012",
  "category": "easy-stationary",
  "notes": "clean baseline frame"
}
```

Optional manual override for mid-move or corrected labels:

```json
{
  "frame_id": "clip_overlay_..._frame0012",
  "category": "mid-move",
  "pieces": [
    {"type": "K", "square": "g1"},
    {"type": "N", "square": null}
  ],
  "notes": "white knight lifted off e2"
}
```

Then materialize the final eval set:

```bash
./.venv/bin/python study/eval/materialize_selection.py
```

That extracts the selected native frames into `study/eval/frames/` and writes `study/eval/labels.jsonl`.

## Metrics

The study eval scripts compute:

- strict piece-set exact match
- placed-board exact match
- per-square accuracy on placed pieces
- per-piece F1 on placed pieces
- category macro averages
