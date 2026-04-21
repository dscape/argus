# Minimal DETR results

## Setup

- Train source: replay-derived real rows (`--source replay`)
- Train subset: `320` rows
- Encoder: frozen DINOv2-base
- Input: board-neighborhood images, `224x224`
- Decoder: `3` layers, `32` queries
- Train run: `outputs/study_final/detr_320_e50_best/`
- Eval run: `outputs/study_final/detr_eval_320_e50_best/`
- Eval set: `study/eval/labels.jsonl` (`344` frames)

## Training curve

Primary validation metric (`placed_board_exact_match`) stayed at `0.0` for all epochs.
Selection therefore fell back to the secondary metric (`per_square_accuracy`).

- Best validation epoch by `(board exact, per-square acc)`: `29`
- Best validation per-square accuracy: `0.8227`
- Best validation board exact: `0.0000`

History: `outputs/study_final/detr_320_e50_best/history.json`

## Held-out eval

| category | count | strict piece exact | placed board exact | per-square acc | piece F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| a-file-rook | 50 | 0.0000 | 0.0000 | 0.5250 | 0.3248 |
| lateral-occlusion | 50 | 0.0000 | 0.0000 | 0.4875 | 0.2124 |
| low-camera-angle | 50 | 0.0000 | 0.0000 | 0.4969 | 0.2493 |
| dense-middlegame | 50 | 0.0000 | 0.0000 | 0.5069 | 0.2813 |
| mid-move | 44 | 0.0000 | 0.0000 | 0.5909 | 0.2724 |
| easy-stationary | 100 | 0.0000 | 0.0000 | 0.4694 | 0.0628 |
| macro | - | 0.0000 | 0.0000 | 0.5128 | 0.2338 |
| overall | 344 | 0.0000 | 0.0000 | 0.5051 | 0.2222 |

## Failure-mode examples

- `outputs/study_final/detr_eval_320_e50_best/failures/mid-move/clip_overlay_h2WrtkfwRl8_clip8_6_frame0006_transient.png`
- `outputs/study_final/detr_eval_320_e50_best/failures/mid-move/clip_overlay_lk82ehiltMI_1_frame0009_transient.png`
- `outputs/study_final/detr_eval_320_e50_best/failures/a-file-rook/clip_overlay_h2WrtkfwRl8_clip8_5_frame0037.png`
- `outputs/study_final/detr_eval_320_e50_best/failures/lateral-occlusion/clip_overlay_h2WrtkfwRl8_clip8_5_frame0397.png`
- `outputs/study_final/detr_eval_320_e50_best/failures/dense-middlegame/clip_overlay_h2WrtkfwRl8_clip8_2_frame0034.png`
- `outputs/study_final/detr_eval_320_e50_best/failures/easy-stationary/clip_overlay_e4lGbQp4pU4_clip70_9_frame0000.png`

## Takeaway

DETR learned substantially more piece identity than the base-head family on the hard categories, especially mid-move, dense middlegame, and lateral occlusion. It still failed the primary exact-match objective because false positives remained too high.
