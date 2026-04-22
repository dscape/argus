# Geometric-mask results

## Setup

- Base checkpoint: `outputs/study_bench/base_head_320/base_head.pt`
- Unmasked eval: `outputs/study_bench/base_head_eval/`
- Masked eval: `outputs/study_bench/base_head_eval_mask/`
- Eval set: `study/eval/labels.jsonl`

## Held-out eval delta vs. base-head

| category | base per-square | mask per-square | Δ per-square | base piece F1 | mask piece F1 | Δ piece F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a-file-rook | 0.5634 | 0.5625 | -0.0009 | 0.0015 | 0.0000 | -0.0015 |
| lateral-occlusion | 0.6209 | 0.6203 | -0.0006 | 0.0017 | 0.0000 | -0.0017 |
| low-camera-angle | 0.6363 | 0.6350 | -0.0012 | 0.0018 | 0.0000 | -0.0018 |
| dense-middlegame | 0.5694 | 0.5684 | -0.0009 | 0.0015 | 0.0000 | -0.0015 |
| mid-move | 0.6332 | 0.6342 | +0.0011 | 0.0000 | 0.0000 | +0.0000 |
| easy-stationary | 0.7344 | 0.7339 | -0.0005 | 0.0016 | 0.0000 | -0.0016 |
| macro | 0.6263 | 0.6257 | -0.0005 | 0.0014 | 0.0000 | -0.0014 |
| overall | 0.6419 | 0.6413 | -0.0005 | 0.0014 | 0.0000 | -0.0014 |

## Failure-mode examples

- `outputs/study_bench/base_head_eval_mask/failures/mid-move/clip_overlay_h2WrtkfwRl8_clip8_6_frame0006_transient.png`
- `outputs/study_bench/base_head_eval_mask/failures/a-file-rook/clip_overlay_h2WrtkfwRl8_clip8_5_frame0037.png`
- `outputs/study_bench/base_head_eval_mask/failures/lateral-occlusion/clip_overlay_h2WrtkfwRl8_clip8_5_frame0397.png`
- `outputs/study_bench/base_head_eval_mask/failures/dense-middlegame/clip_overlay_h2WrtkfwRl8_clip8_2_frame0034.png`

## Takeaway

Inference-only masking did not rescue the weak base-head model. The delta is effectively zero and slightly negative on every category except a negligible mid-move per-square bump.
