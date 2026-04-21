# Base-head results

## Setup

- Train source: replay-derived real rows (`--source replay`)
- Train subset: `320` rows
- Encoder: frozen DINOv2-base
- Input: projected piece crops, `224x224`
- Train run: `outputs/study_bench/base_head_320/`
- Eval run: `outputs/study_bench/base_head_eval/`
- Eval set: `study/eval/labels.jsonl` (`344` frames: `44` mid-move available, not `50`)

## Training curve

Only one epoch was practical for this architecture on local hardware.

| epoch | train loss | train decision acc | val loss | val type acc | val base acc | val decision acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.7680 | 0.8891 | 3.7603 | 0.5076 | 0.5796 | 0.5080 |

History: `outputs/study_bench/base_head_320/history.json`

## Held-out eval

| category | count | strict piece exact | placed board exact | per-square acc | piece F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| a-file-rook | 50 | 0.0000 | 0.0000 | 0.5634 | 0.0015 |
| lateral-occlusion | 50 | 0.0000 | 0.0000 | 0.6209 | 0.0017 |
| low-camera-angle | 50 | 0.0000 | 0.0000 | 0.6363 | 0.0018 |
| dense-middlegame | 50 | 0.0000 | 0.0000 | 0.5694 | 0.0015 |
| mid-move | 44 | 0.0000 | 0.0000 | 0.6332 | 0.0000 |
| easy-stationary | 100 | 0.0000 | 0.0000 | 0.7344 | 0.0016 |
| macro | - | 0.0000 | 0.0000 | 0.6263 | 0.0014 |
| overall | 344 | 0.0000 | 0.0000 | 0.6419 | 0.0014 |

## Failure-mode examples

- `outputs/study_bench/base_head_eval/failures/mid-move/clip_overlay_h2WrtkfwRl8_clip8_6_frame0006_transient.png`
- `outputs/study_bench/base_head_eval/failures/mid-move/clip_overlay_lk82ehiltMI_1_frame0009_transient.png`
- `outputs/study_bench/base_head_eval/failures/a-file-rook/clip_overlay_h2WrtkfwRl8_clip8_5_frame0037.png`
- `outputs/study_bench/base_head_eval/failures/lateral-occlusion/clip_overlay_h2WrtkfwRl8_clip8_5_frame0397.png`
- `outputs/study_bench/base_head_eval/failures/dense-middlegame/clip_overlay_h2WrtkfwRl8_clip8_2_frame0034.png`
- `outputs/study_bench/base_head_eval/failures/easy-stationary/clip_overlay_e4lGbQp4pU4_clip70_9_frame0000.png`

## Takeaway

Under the feasible local training budget, the base-head variant did not reach non-zero exact-match on any category and barely learned piece identity at all.
