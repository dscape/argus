# Template matching v2 study

This study keeps the v1 core idea intact:

- frozen backbone
- per-tournament template bank
- cosine-similarity classification

It changes how the bank and proposal embeddings are built.

## Goal

Beat the v1 template pipeline, especially on tournaments where starting-position crops are visually indistinct or contaminated by occluders and background.

Baseline references:

- `study/templates/RESULTS.md`
- `study/FINAL_DECISION.md`

## Three mechanisms

### 1. Geometric visibility gating

For each frame, use board geometry, FEN, and camera calibration to rasterize all 32 piece cuboids into image space. For each target piece, keep only the pixels that belong to that piece after subtracting all closer occluders. Downsample that visibility into the DINO patch grid and use it as a patch-token mask before pooling.

Intended effect: stop clocks, bottles, hands, neighboring pieces, and hidden piece regions from becoming part of a class identity.

### 2. Multi-frame aggregation across the whole game

Build each tournament bank from hundreds or thousands of PGN-aligned frames instead of a handful of starting-position crops. Use visibility-weighted aggregation so cleaner middlegame views can repair weak opening templates.

Intended effect: templates reflect the piece class, not one bad viewpoint.

### 3. Template quality gate

Before evaluation, reject tournament banks that fail visible-distinguishability checks.

Intended effect: do not run expensive evals on banks that are already visibly collapsed.

## Directory layout

- `geometry/` — analytic cuboid visibility masks and debug renders
- `isolation/` — PCA and geometric foreground masking experiments
- `builder/` — multi-frame bank construction and quality gating
- `inference/` — masked embedding and template matching
- `eval/` — evaluation runs and failure inspection artifacts
- `data/` — built banks, previews, and quality reports
- `proposals/` — thin adapters for proposal sources reused from `study/templates/proposals/`

## Constraints

- Core Argus stays read-only.
- All v2 work lives under `study/templates-v2/`.
- Do not read or copy v1 implementation code.
- Reuse proposal wrappers later by importing from `study/templates/proposals/`, not by copying.

## Execution plan

1. Scaffold only.
2. Build geometric visibility masks.
3. Add foreground-isolation methods.
4. Add masked patch-token embedding.
5. Build multi-frame tournament banks.
6. Gate banks on distinguishability.
7. Add the v2 classifier.
8. Reuse proposal sources.
9. Extend the eval harness.
10. Run the known-good tournament first.
11. Run the hard failure tournament.
12. Produce failure-inspection sheets before deciding next steps.

Current state: scaffold only. No v2 code has been added yet.
