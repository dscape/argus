# Physical pipeline plan

## Why this exists

Physical-board vision is a separate problem from 2D overlay reading. Overlay models stay in `pipeline/overlay/`, physical-board models stay in `pipeline/physical/`, and the only shared contract is in `pipeline/shared/` once pixels have been abstracted into board-state observations. Temporal fusion is explicitly shareable through `BoardObservation` (`fen`, `square_confidences`, `timestamp_seconds`).

## Reading-derived execution rules

- **DINOv2 eval recipe:** start each visual subtask with **frozen features + linear probe / shallow head** before any end-to-end fine-tuning. If the frozen-feature baseline is bad, fix data, labels, crops, and evaluation first.
- **Karpathy recipe:** inspect raw examples first, disable augmentation initially, overfit a tiny batch, verify loss/metric baselines, and only add complexity after a simple model fits the task.
- **No more cross-domain leakage:** overlay classifiers are forbidden on physical camera crops.
- **Held-out real eval first:** no physical-board model training starts before `data/physical/eval/` exists and is excluded by source video id from all training splits.

## Status board

- [x] Directory split started: `pipeline/physical/`, `pipeline/shared/`, local folder-level `AGENTS.md`
- [x] Shared `BoardObservation` contract defined
- [x] Held-out physical square eval set populated with manually labeled real square crops
- [ ] Frame classification trained and validated
- [ ] Physical board localization trained and validated
- [ ] Physical board rectification/corner refinement trained and validated
- [ ] Per-square physical state model trained and validated
- [ ] Temporal fusion revalidated on physical observations
- [ ] Raw-video-to-PGN physical pipeline benchmarked end-to-end

## Step 0 — shared contract + held-out eval data

### Spec 0A: Shared board observation contract

| Field | Value |
| --- | --- |
| Input | Any per-frame board reader output after pixels have been converted to a board hypothesis |
| Output | `BoardObservation(fen, square_confidences[64], timestamp_seconds, source)` |
| Loss | N/A |
| Eval metric | Interface adoption: temporal fusion consumes only `BoardObservation`, never raw images |
| Success | Overlay and physical readers can both emit the same observation type; fusion code is source-agnostic |
| Secondary | Store per-square confidence, not just whole-board confidence, so fusion can reason about uncertainty locally |

### Spec 0B: Held-out physical square eval set

Current snapshot on this branch:
- `846` annotated rectified board frames
- `54,142` labeled square crops
- `4` held-out source videos
- held-out source video ids are now excluded structurally by `python -m pipeline physical-split-clips`


| Field | Value |
| --- | --- |
| Input | Real broadcast camera crops of physical boards, manually corner-rectified and square-labeled via dev-tools |
| Output | `data/physical/eval/board_annotations.jsonl`, `data/physical/eval/square_manifest.jsonl`, rectified boards, square crops |
| Loss | N/A |
| Eval metric | Dataset coverage: number of labeled square crops, source-video count, class coverage, zero overlap with training video ids |
| Success | At least **300** hand-labeled square crops from **10+** held-out source videos, with saved board-corner annotations and no training/eval video overlap |
| Secondary | Every class present; at least **20** examples for each non-empty class; all labels trace back to source clip + frame index |

## Step 1 — frame classification

### Spec 1: Broadcast frame classification

| Field | Value |
| --- | --- |
| Input | Full video frame `(3, H, W)` from broadcast footage |
| Output | Frame-type probabilities over `{board_view, overlay, player_closeup, crowd, other}` |
| Loss | Cross-entropy on frame label |
| Eval metric | Macro-F1, `board_view` precision/recall, calibration error |
| Success | `board_view` recall **> 0.97** at precision **> 0.95** on held-out real frames; macro-F1 **> 0.90** |
| Secondary | Frozen DINOv2 + linear probe is the first baseline; inference under **15 ms/frame** on dev hardware |

### Notes

- Start with a frozen DINOv2 encoder and linear head.
- Build a dumb baseline first (`always board_view`, `always non-board`) to make sure metrics are honest.
- Overfit a tiny labeled subset before scaling.

## Step 2 — board localization

### Spec 2A: Physical board detection

| Field | Value |
| --- | --- |
| Input | Full frame already classified as `board_view` |
| Output | One physical-board detection `(bbox, confidence)` |
| Loss | Standard detector loss (objectness + box regression; e.g. YOLO/DETR-style box loss) |
| Eval metric | mAP@0.5 and recall@confidence threshold |
| Success | mAP@0.5 **> 0.90**, recall **> 0.97** at deployment threshold on held-out real frames |
| Secondary | Robust across zoom changes and partial occlusions; low false positives on non-board frames |

### Spec 2B: Corner / homography refinement

| Field | Value |
| --- | --- |
| Input | Board crop from Step 2A |
| Output | Four ordered board corners `(tl, tr, br, bl)` and a rectifying homography |
| Loss | Corner L1 / smooth-L1 on normalized coordinates + reprojection loss after rectification |
| Eval metric | Mean corner error (pixels and normalized), IoU after warped board mask, downstream rectified-square alignment score |
| Success | Mean corner error **< 3%** of board width on held-out real frames; rectified boards are aligned enough that manual square labels land on intended squares |
| Secondary | Corner annotations collected for eval-set boards become the seed supervision for this model |

### Notes

- Existing bbox labels are enough to start 2A immediately.
- 2B needs explicit corner labels; the physical-square annotation UI is also the corner-labeling tool.
- Do **not** merge 2A and 2B until the simple two-stage baseline is understood.

## Step 3 — per-square state

### Spec 3: Physical board square-state classification

| Field | Value |
| --- | --- |
| Input | Rectified physical board image, typically `384×384` to `512×512` RGB |
| Output | `64 × 13` square logits and a derived `BoardObservation` |
| Loss | Sum / mean of 64 per-square cross-entropies |
| Eval metric | Overall square accuracy, macro-F1, non-empty-only accuracy, per-class confusion matrix, whole-board exact match on labeled boards |
| Success | Overall square accuracy **> 0.90**, macro-F1 **> 0.75**, non-empty-only accuracy **> 0.80** on the held-out real square-crop eval set |
| Secondary | Well-calibrated per-square confidences (ECE **< 0.05**), whole-board exact match improves enough for fusion to see real signal |

### Notes

- First baseline: frozen DINOv2 features on square crops + linear probe / shallow MLP.
- Current result from the independent square-crop baseline on this branch: it transfers very poorly from synthetic physical square crops to held-out real broadcast squares, so it should be treated as a failed diagnostic rather than a deployable reader.
- Board context helps somewhat, but the synthetic source also has to match the target: the real eval boards are **rectified** physical boards, so the next attempt needs rectified synthetic renders with realistic oblique 3D piece appearance.
- A later DINO-vs-YOLO sweep on rectified top-down synthetic boards confirmed that backbone choice is secondary to the domain gap:
  - DINO preserves more non-empty piece signal.
  - YOLO can look better on overall square accuracy only by collapsing toward `empty`, so non-empty accuracy and macro-F1 must remain primary diagnostics.
- Saved sample boards from the sweep show an additional mismatch that was easy to miss from metrics alone: real rectified boards often contain strong directional blur / resampling streaks on pieces.
- A follow-up artifact pass that added higher-resolution renders, piece-only strip-wise distortion, anisotropic resampling, and directional blur improved transfer somewhat, but the task is still far from solved.
  - Current best run: `outputs/2026-04-12/physical_board_probe_dino_topdown_aug_weighted_v2/`
  - held-out real square accuracy: `0.3151`
  - held-out real non-empty accuracy: `0.2466`
  - held-out real macro F1: `0.1064`
- A higher-input-size DINO check at `336×336` did not help on the current synthetic source, so resolution alone is not the bottleneck.
- Replay-derived non-held-out STL Chess Club boards can now be rectified automatically via transferred channel-level corner templates, and the refined export lives under `outputs/2026-04-12/physical_real_board_dataset_export_refined/`.
- Adding a bounded per-clip corner-refinement step materially improved those pseudo-real boards, and mixing them into training now helps more than the earlier raw transfer did.
- A major hidden bug was later found in the replay-derived labels: real clips store absolute sampled `frame_indices`, but the pseudo-real label builder had been comparing `move_frame_indices` against clip-local indices. Fixing that bug made the replay labels advance correctly with moves and increased the corrected stride-4 pseudo-real pool to `658` boards.
- The original shared linear board readout also turned out to be a genuine bottleneck. Frozen-DINO transfer improves a lot once the head is allowed to use explicit square position and mild cross-square context:
  - `head_type=pos_mlp`: learned square-position embeddings + MLP readout
  - `head_type=transformer`: learned square-position embeddings + shallow transformer over the 64 square tokens
- Single-layer DINO features were also leaving signal on the table. The current best frozen-feature setup averages DINO hidden states from layers `8,10,11` before the board head.
- The best new single checkpoints on corrected pseudo-real data are now:
  - transformer + layers `8,10,11`: `physical_board_probe_dino_topdown_transformer_real658_fixed_layers8_10_11_seed3`
    - square accuracy: `0.5341`
    - non-empty accuracy: `0.3843`
    - macro F1: `0.2758`
  - positional MLP + layers `8,10,11`: `physical_board_probe_dino_topdown_posmlp512_real658_fixed_layers8_10_11_rw4_seed0`
    - square accuracy: `0.4611`
    - non-empty accuracy: `0.4103`
    - macro F1: `0.2980`
- Weight-space checkpoint averaging is still the wrong way to ensemble these newer heads because it drifts back toward empty-heavy behavior.
- The current committed runtime candidate is therefore still a **logit-space** ensemble under `weights/physical/`, but it has now changed shape again after adding pseudo-real source-video validation for checkpoint selection, a lightweight shared runtime postprocess, and clip-ordered runtime selection:
  - code version: `v7`
  - stateless-crop reference artifact: `weights/physical/v6r2.pt`
  - deployed runtime artifact: `weights/physical/v7r4.pt`
  - shared DINO layers: `8,10,11`
  - deployed runtime members:
    - `physical_board_probe_dino_topdown_posmlp512_real_train_split_holdoutpsr_layers8_10_11_rw4_seed0`
    - `physical_board_probe_dino_topdown_posmlp512_real658_fixed_layers8_10_11_rw4_seed0`
    - `physical_board_probe_dino_topdown_transformer_real658_fixed_layers8_10_11_seed0`
  - deployed ensemble weights: `21,7,1`
  - runtime postprocess:
    - back-rank pawns are reassigned to the best non-pawn class
    - each color is forced to have **exactly one** king by removing duplicate kings and inserting a king only when one is missing
  - runtime metadata now also records the recommended adaptive temporal smoothing config for the stateful reader
- Relative to `v5r2`, the best committed physical reader path is clearly better on the diagnostics that actually matter for physical board reading (`non-empty` and `macro-F1`) even though board exact match is still `0.0`.
- A quick temporal-stability diagnostic on the held-out physical clips showed that the current reader is flickering far more than the real boards (`13.45%` predicted square-flip rate vs `0.25%` ground truth), so lightweight sequence smoothing is now justified rather than speculative.
- The repo therefore now also has a stronger temporal baseline on top of the same runtime weights:
  - shared helper: `pipeline/shared/board_smoothing.py`
  - stateful runtime reader: `pipeline.physical.square_classifier.PhysicalBoardSequenceReader`
  - hybrid full-frame runtime now keeps an **adaptive** EMA over physical board logits across consecutive physical frames
  - current recommended temporal config from runtime metadata:
    - mode: `adaptive_ema`
    - low alpha: `0.02`
    - high alpha: `0.12`
    - change threshold: `8`
  - current best deployed clip-ordered eval: `outputs/2026-04-13/physical_runtime_eval_v22_temporal_adaptive_v7r4.json`
    - square accuracy: `0.5259`
    - non-empty accuracy: `0.4667`
    - macro F1: `0.3700`
  - current best stateless single-frame eval remains `v6r2`: `outputs/2026-04-13/physical_runtime_eval_v11_exact_kings_runtime.json`
    - square accuracy: `0.5115`
    - non-empty accuracy: `0.4427`
    - macro F1: `0.3433`
  - a quick greedy legal-move candidate filter did not beat smoothing, and naive pseudo-real agreement filtering regressed badly, so neither was kept
- The repo now also has two pieces of infrastructure for the next data-centric step:
  - manual non-held-out physical board labels can be collected under `data/physical/train_manual/`
  - pseudo-real source-video holdouts can be used during training for checkpoint selection via `scripts/train_physical_board_probe.py --real-val-source-videos ... --selection-metric auto`
- The practical next attempt should still prioritize **manual non-held-out real supervision** and better localization / corner quality before more backbone work, but pseudo-real real-val checkpoint selection plus lightweight board-state constraints are now real levers rather than just hypotheses.
- Empty squares will dominate; success is not allowed to hide behind empty-square accuracy.
- Synthetic physical renders are training fuel, not validation data.

## Step 4 — temporal fusion

### Spec 4: Source-agnostic legal-game reconstruction

| Field | Value |
| --- | --- |
| Input | Time-ordered stream of `BoardObservation` items from overlay or physical readers |
| Output | Legal move sequence, reconstructed PGN, and per-move confidence |
| Loss | Sequence negative log-likelihood / path score under the observation model (if trainable); otherwise tune against held-out move accuracy and legality metrics |
| Eval metric | Move accuracy, legal-game rate, PGN edit distance, move-timestamp latency |
| Success | On a held-out noisy-observation benchmark, legal-game rate **> 0.98** and move accuracy **> 0.95**; on held-out real physical sequences, move accuracy **> 0.80** once Step 3 succeeds |
| Secondary | Graceful recovery from missed reads, stable confidence estimates, explicit ablations by upstream observation quality |

### Notes

- Validate fusion in isolation with injected noise before blaming it for upstream perception failures.
- Fusion only becomes the bottleneck after Step 3 stops being the bottleneck.

## Step 5 — end-to-end physical pipeline benchmark

### Spec 5: Raw video to legal PGN

| Field | Value |
| --- | --- |
| Input | Broadcast video clip containing a physical board view |
| Output | Legal PGN plus debugging artifacts: frame classes, board boxes, corners, rectified boards, square predictions, fused moves |
| Loss | N/A (integration benchmark) |
| Eval metric | Clip-level move accuracy, legal-game rate, board-read coverage, failure attribution by stage |
| Success | End-to-end benchmark runs without manual intervention on held-out clips and produces a mostly correct legal game; every failure is attributable to a named stage, not a mixed abstraction boundary |
| Secondary | Reproducible benchmark command, saved diagnostics in `outputs/`, regression tracking over time |

## Immediate execution order

1. Finish the structural split and keep the invariant enforced in code review.
2. Populate the held-out physical eval set via dev-tools; no model training before this exists.
3. Export physical train/val data with `python -m pipeline physical-split-clips` so held-out eval source videos are excluded automatically.
4. Train the simplest possible frame classifier baseline.
4. Train board detection, then corner refinement.
5. Train per-square classification on rectified physical boards.
6. Re-run temporal fusion with `BoardObservation` inputs from the physical reader.
7. Benchmark raw-video-to-PGN on held-out physical clips.

## Diagnostic checklist per model step

- Inspect raw inputs and labels by hand.
- Save sample images from both synthetic and held-out real datasets into `outputs/` so domain mismatch is visible, not just inferred from metrics.
- Compute trivial baselines first.
- Overfit one tiny batch.
- Disable augmentation for the first honest run.
- Compare DINO vs YOLO on the same data before assuming the issue is the backbone.
- Track both train and held-out metrics from the start.
- Never trust overall square accuracy without non-empty accuracy and macro-F1.
- Save failure cases and confusion matrices into `outputs/`.
- Do not move to the next step while the current step is still information-starved.
