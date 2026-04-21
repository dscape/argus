# Template-matching study

This study explores a non-head piece recognition path under `study/templates/`.

## Goal

Instead of training a piece classifier head, build a per-tournament template bank from known standard-starting-position frames, embed each crop with the existing frozen vision backbone, and classify later piece crops by nearest-template cosine similarity.

## Scope

- `builder/`: template-bank construction from starting-position frames
- `inference/`: embedding + template-match classification
- `eval/`: end-to-end evaluation harness and run outputs
- `data/`: serialized template banks, previews, and other generated study artifacts
- `proposals/`: piece-crop proposal wrappers and previews

Core Argus stays read-only. All experimental code for this path lives under `study/templates/`.

## Existing eval set

Reuse the held-out eval set under `study/eval/`:

- schema and workflow: `study/eval/README.md`
- frames: `study/eval/frames/`
- labels: `study/eval/labels.jsonl`

## Comparison target

Report results in the same schema used by the earlier studies so this path can be compared directly against:

- `study/base-head/RESULTS.md`
- `study/detr-minimal/RESULTS.md`
- `study/FINAL_DECISION.md`
