# Final decision

## Eval set

Materialized eval set:

- labels: `study/eval/labels.jsonl`
- frames: `study/eval/frames/`

Actual breakdown is `344` frames, not the planned `350`, because the held-out transient annotations only exposed `44` usable mid-move frames:

- `50` a-file-rook
- `50` lateral-occlusion
- `50` low-camera-angle
- `50` dense-middlegame
- `44` mid-move
- `100` easy-stationary

Selection summary: `study/eval/selection_summary.json`

## Side-by-side comparison

### Primary metrics

All three configurations stayed at `0.0000` on both:

- strict piece exact
- placed board exact

That means the primary decision rule did **not** separate the models under the feasible local training budget.

### Secondary metrics

| variant | macro per-square | macro piece F1 |
| --- | ---: | ---: |
| base-head | 0.6263 | 0.0014 |
| base-head + mask | 0.6257 | 0.0000 |
| minimal DETR | 0.5128 | 0.2338 |

Hard-category piece F1 macro:

| category | base-head | base-head + mask | minimal DETR |
| --- | ---: | ---: | ---: |
| lateral-occlusion | 0.0017 | 0.0000 | 0.2124 |
| low-camera-angle | 0.0018 | 0.0000 | 0.2493 |
| dense-middlegame | 0.0015 | 0.0000 | 0.2813 |
| mid-move | 0.0000 | 0.0000 | 0.2724 |

## Interpretation

1. **Mask is not worth shipping from this study.**
   - It produced effectively zero gain.
   - On the measured metrics it was flat to slightly worse.

2. **DETR shows a real representational signal.**
   - On every hard category, DETR's piece-F1 is dramatically above the base-head family.
   - The biggest gap is mid-move, where the per-square family is structurally weak.

3. **But DETR still does not win the actual objective yet.**
   - Primary exact-match metrics are flat zero.
   - DETR improves piece identity, but not enough to reconstruct whole boards cleanly.
   - False positives / duplicate pieces remain too high.

## Decision

**Do not proceed to a full rewrite yet.**

The current evidence is:

- too weak to justify shipping the base-head + mask path
- suggestive enough that DETR is the more promising architecture
- not strong enough on the primary metric to green-light a rewrite

So the result is **"promising but undertrained"**, not **"rewrite now"** and not **"ship the mask"**.

## Recommended next step

If this continues, the next project should be:

1. train DETR on a much larger real broadcast set
2. tune decode calibration / presence thresholding / duplicate suppression
3. rerun the same held-out eval before adding chess constraints or temporal layers

That keeps the architectural question clean.
