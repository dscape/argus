# Memory log

## 2026-04-16T18:59:43.849Z | experiment | low | Added physical crop types comparison attachment

Created `.agents/memory/attachments/physical_crop_types_example.png`, a user-facing 6-panel visual showing one held-out physical example through: source frame, native board-neighborhood crop, oblique board crop, rectified board, rectified 64-tile split, and oblique square-context crops. Useful for explaining the repo's crop families and how rectified vs oblique representations differ.

## 2026-04-16T19:25:49.017Z | prompt | high | Diagnosis-first plan for pushing board_exact toward 90%

User pointed to `outputs/plan.md`, which sets the objective to push `board_exact` as close to 90% as the current data and architecture support on the evaluated target split. Key constraints: state current baseline before changes; first run a failure study on the best rectified+lookahead run; sample 100 failure episodes (first failing frame + ~10 preceding frames, capped per video); tag failures into rectification/localization, piece-classifier/square-evidence, temporal ambiguity, decoder/legal-hypothesis/error-propagation, eval/label issue, or other; build a viewer to scrub leading frames for each tagged failure; choose subsequent experiments only based on those buckets; avoid architecture sweeps without failure-mode justification; abandon low-return directions; use oracle if experiment results contradict hypotheses or when failure data does not disambiguate.

## 2026-04-16T19:45:47.597Z | decision | medium | Added dev-tools physical failure-study viewer and tagging UI

Implemented a new Evaluate > Failures dev-tools page backed by `/api/evaluate/physical-failures/*` endpoints. The viewer auto-discovers failure-study bundles under `outputs/`, loads `summary.json` + `manifest.json` + `manual_buckets.csv`, shows failure metadata and baseline eval metrics, scrubs leading raw frames for a selected failure entry, displays the rendered anchor diagnostics panel, and saves `final_bucket`/`notes` edits back to `manual_buckets.csv`. Added backend service coverage in `tests/dev_tools/test_physical_failure_study_service.py`.

## 2026-04-16T20:17:45.528Z | experiment | high | Episode-based failure study on rectified realplusmanual baseline

Updated `physical-board-failure-study` to sample failure episodes rather than raw failing frames, with preceding-frame context and per-video caps. Regenerated the study for `outputs/2026-04-16/tracker_sweep_rectified_realplusmanual/eval_w2_m10.json` into `outputs/2026-04-16/physical_board_failure_study_rectified_realplusmanual_w2_m10/`. Result: 14 total episodes, 13 selected under `max_per_video=5`. Tagged selected episodes: 12 `temporal in-between / move execution ambiguity`, 1 `piece classifier / square evidence`. Dominant failure mode is now temporal ambiguity around move execution, often followed by long desync episodes when the decoder misses the transition frame.

## 2026-04-16T20:25:49.902Z | decision | medium | Failure-study viewer needs outputs mounted into dev-tools API

Diagnosed empty `/evaluate/failures` page: the dev-tools API container returned `{"studies": []}` because `docker-compose.yaml` did not mount `./outputs` into `/app/outputs`, so `physical_failure_study_service` could not discover generated bundles. Fixed by adding the outputs volume mount and aligning `dev-tools/api/routers/evaluate/physical_failure_study.py` with the episode-based service/frontend (`episode_id` updates plus `/image` and `/export-csv` endpoints). Container must be recreated for the new volume mount to take effect.

## 2026-04-16T20:38:56.496Z | decision | medium | Failure viewer now renders board triad and tagging guidance

Updated `/evaluate/failures` to render crop + ground-truth/stateless/decoded chess boards inside the scrubber using the same failure-study color semantics as the generated panels. Also added bucket descriptions, an episode-level heuristic read, and legal-candidate summaries so tagging focuses on the first FAIL frame instead of the later desync tail. Frontend now consumes `stateless_error_squares` from the failure-study service for proper stateless board highlighting.

## 2026-04-16T21:05:00.816Z | decision | medium | Local authresearch decoder workspace scaffolded for overnight Argus experiments

Created a local gitignored `authresearch/` workspace that mirrors Karpathy autoresearch's expected shape (`prepare.py`, `train.py`, `program.md`, `pyproject.toml`, `results.tsv`) but is adapted to Argus and the current Mac/MPS environment.

Key choices:
- Scoped the workspace to the highest-ceiling current direction: decoder-only work on the rectified realplusmanual baseline, because the latest tagged failure study is 12/13 `temporal in-between / move execution ambiguity` and only 1/13 classifier.
- `authresearch/prepare.py` is fixed infrastructure: it caches held-out rectified-board logits from `outputs/2026-04-16/tracker_sweep_rectified_realplusmanual/eval_w2_m10.json` / its `board_probe.pt`, and verifies the cache reproduces the baseline exactly.
- `authresearch/train.py` is the editable experiment surface: baseline is the current lookahead tracker (`w2 m10`), but the file also exposes a segmental decoder path for temporal/dwell experiments.
- Runs auto-snapshot the current `train.py`, write JSON reports, append `results.tsv`, update `best_train.py` on keep, and restore `train.py` from `best_train.py` on discard.

Validation done locally:
- `.venv/bin/python authresearch/prepare.py` succeeded and verified `board_exact=0.216568` against the fixed baseline.
- `.venv/bin/python authresearch/train.py` succeeded and logged the baseline run.

Start commands from repo root:
- prep once: `.venv/bin/python authresearch/prepare.py`
- run one experiment: `.venv/bin/python authresearch/train.py > authresearch/run.log 2>&1`
- for Pi, prompt it to read `authresearch/program.md` and start the loop.

Environment note remains unchanged: duplicate `AVFFrameReceiver` / `AVFAudioReceiver` implementations from `cv2` and `av` still print on startup, but did not block prepare/train runs.

## 2026-04-16T22:06:27.573Z | experiment | high | Authresearch score-gated hybrid decoder beats rectified lookahead baseline

In `authresearch/`, a hybrid decoder that runs the baseline lookahead (`w2,m10`) and the best board-only segmental candidate (`top_board_candidates=1`, `move_score_margin=10`, nearby secondary proposals) per sequence, then switches to segmental only when it adds moves **and** gains at least `200` total board-logprob with at least `200` board-logprob gain per added move, improved the fixed cached eval from `board_exact=0.216568` to `0.230769`.

Run: `authresearch/runs/20260416T220240Z_hybrid_lookahead_segmental_scoregain_v1.json`

Metrics:
- `board_exact`: `0.230769` (+`0.014201`)
- `static_false_change_rate`: `0.020699`
- `move_detection_recall`: `0.468750`
- `macro_f1`: `0.841499`
- `non_empty_accuracy`: `0.868121`

The gating selected segmental only on two sequences with very large self-score gains per extra move (`clip70_9` and `clip8_5`), avoiding the clips where full-segmental over-committed despite higher raw board score. This suggests the next decoder work should stay hybrid/score-gated rather than replacing lookahead globally with segmental.

## 2026-04-16T22:15:03.417Z | experiment | high | Authresearch hybrid follow-ups plateau; state-aware fallback times out

Follow-up experiments after the score-gated hybrid improvement did not beat `board_exact=0.230769`.

Key findings:
- Relaxing the hybrid score-gain gates (`total150`, `permove150`) produced identical metrics to the original hybrid, so those thresholds were not the active limiter.
- Adding `state_aware_proposal_passes=1` to the segmental fallback timed out twice under the 120s authresearch budget, both standalone and inside the hybrid fallback path.
- A broader manual search over alternate segmental fallback variants/pools (`drop1`, `sep5`, `top8`, `r2`, non-secondary baseline) did not beat the current hybrid under the fixed tie-break order; the best overall result stayed the existing score-gated hybrid family.

Practical conclusion: the cheap whole-sequence score-gated hybrid is the current local optimum in this workspace. Further gains likely require a different repair granularity (e.g. segment-level/suffix-level gating) or a slower fallback search outside the current authresearch budget.

## 2026-04-16T23:02:37.338Z | experiment | high | Measured state-aware hybrid fallback runtime at ~276s

Ran the prepared `authresearch` experiment `hybrid_lookahead_segmental_state1_v4` with `SEGMENTAL_CONFIG["state_aware_proposal_passes"] = 1` under a longer manual timeout to measure real runtime instead of relying on the 120s controller cap.

Result:
- report: `authresearch/runs/20260416T225626Z_hybrid_lookahead_segmental_state1_v4.json`
- wall time: about `280s`
- reported `elapsed_seconds`: `275.67`
- metrics: `board_exact=0.230769`, `move_detection_recall=0.468750`, `static_false_change_rate=0.021992`

So one state-aware proposal pass is not a cheap-loop fit: it needs about 4.6 minutes on this cached eval and did not improve `board_exact` over the existing score-gated hybrid; it slightly worsened false-change rate. After measurement, `authresearch/train.py`, `best_train.py`, and `best_result.json` were restored to the better `...scoregain_permove150_v3` baseline state.

## 2026-04-16T23:07:52.119Z | decision | medium | Failures viewer separates raw snapshot from optional model crop

Updated the physical failure-study viewer and API so the main comparison uses equal-sized panels for raw snapshot, optional model crop, GT, stateless, and decoded boards. The service now lazily materializes pre-rectification clip-frame snapshots into `outputs/.../viewer_cache/raw_snapshots/` keyed by `annotation_id`, exposes `raw_image_path` + optional `processed_image_path`, and the thumbnail scrubber prefers the raw snapshot for human review. Chose clip-frame snapshots instead of source-video extraction for robustness and zero extra dependency on downloaded native videos.

## 2026-04-16T23:08:33.442Z | decision | high | Authresearch now has separate cheap and slow controllers

Added a separate slow-lane controller for `authresearch/`.

Files:
- `authresearch/controller.sh` — still the main 120s cheap loop
- `authresearch/controller_slow.sh` — new 360s off-budget lane for state-aware experiments

Implementation details:
- `controller.sh` now takes lane settings from environment so the timeout, run log, session file, stop file, iteration file, and prompt guidance can vary by lane without duplicating the core loop.
- `controller_slow.sh` is a thin wrapper that points the shared controller at `run_slow.log`, `STOP_SLOW`, `controller_slow.iteration`, a separate session file, and 360s timeout/prompt guidance.
- `authresearch/program.md` documents both lanes and notes that they should not be run simultaneously because they share `train.py`.

Validated with `bash -n`, smoke runs for both controllers, plus `make typecheck`, `make lint`, and `make test`.

## 2026-04-16T23:32:07.606Z | decision | high | Prepared separate whole-board authresearch workspace on native board-neighborhood crops

Added a separate local workspace at `authresearch_wholeboard/` for whole-board experiments that use the raw native board-neighborhood crop (`native_image_bbox`) rather than rectified boards or square crops.

Key files:
- `authresearch_wholeboard/prepare.py`
- `authresearch_wholeboard/train.py`
- `authresearch_wholeboard/best_train.py`
- `authresearch_wholeboard/program.md`
- `authresearch_wholeboard/controller.sh`

Design choices:
- Physical inputs are fixed to the raw native board-neighborhood crop with no red lines / board-quad overlays baked into the model input.
- The workspace reuses the existing whole-board direct reader path (`scripts/prepare_direct_board_reader_dataset.py` and `scripts/train_direct_board_reader.py`) instead of the rectified-board decoder cache used by `authresearch/`.
- Whole-board ranking uses physical tracker `board_exact` first, then lower static false-change, higher move recall, higher tracker macro-F1, and higher tracker non-empty accuracy.
- Baseline reference is the existing DINO whole-board smoke run at `outputs/2026-04-16/direct_board_reader_dinov2_blenderclips_smoke1/summary.json`.
- The new controller is independent from `authresearch/`, so both workspaces can run in parallel without sharing `train.py` or controller state.

## 2026-04-16T23:51:55.138Z | decision | high | Whole-board lane now carries previous-board context and supports optional state-conditioned readers

Extended the whole-board direct-reader path so prepared samples can include prior board context and train-time model variants can consume it.

What changed:
- `scripts/prepare_direct_board_reader_dataset.py` now writes `previous_labels`, `previous_board_available`, and `previous_side_to_move` into prepared manifests when the prior board is known.
- For physical clips, prior board context is seeded from `initial_board_fen` / `initial_side_to_move` and then advanced frame-by-frame from ground-truth board changes.
- For synthetic clips with FEN history, prior board context is derived from prior clip FENs / initial FEN when present.
- ChessReD remains prior-context unavailable.
- `pipeline/physical/direct_board_reader_data.py` now exposes previous-board context in dataset samples.
- `pipeline/physical/direct_board_reader.py` and `scripts/train_direct_board_reader.py` now support optional previous-board conditioning modes (`none`, `add`, `gated`) plus optional previous side-to-move conditioning.
- Validation for conditioned models avoids teacher-forcing leakage by using an autoregressive prediction path rather than ground-truth previous boards.

Current default `authresearch_wholeboard/train.py` baseline still leaves `PREVIOUS_BOARD_CONDITIONING = "none"`; the lane is now capable of state-conditioned variants, but they are not forced on by default.

## 2026-04-17T07:11:08.040Z | prompt | high | Add temporal-transient physical annotation labels

User diagnosis: dominant error is temporal move-execution ambiguity and the tracker needs an explicit transient-state sequence model that handles piece lift/capture/hand occlusion/placement/settle, precise move timing, and fast recovery after a missed move. First requested step is to update the dev-tool physical annotation flow so clips can be labeled with move start, move end/settled frame, capture vs non-capture, and hand-occluded spans.

## 2026-04-17T07:28:47.681Z | decision | high | Add transient physical clip labels to annotation tool

Extended the physical annotation flow to persist clip-level transient labels per split in `data/physical/{train,val}/transient_annotations.jsonl`. Added API endpoints to load/save/delete per-clip transient annotations, dataset helpers for move timing and hand-occlusion spans, a new `PhysicalTransientAnnotationPanel` in the physical annotation UI, and tests covering transient save/load/delete for eval and manual-train datasets. Labels cover per-move start frame, settled frame, capture flag, and hand-occluded spans.

## 2026-04-17T09:16:44.890Z | decision | high | Simplify physical transient labeling UX around move list

Reworked the physical transient-label UI so it no longer uses a separate full panel or manual capture labeling. The existing move list is now the primary labeling surface: each move shows `!` until both touch start and touch end are set, then displays `fstart > freplay > fend`. Two buttons set touch start/end for the active move at the current frame. Hand occlusion is toggled directly on the source frame image, while persistence still uses split-local `transient_annotations.jsonl`. The clip index checkmark now reflects completed transient move coverage instead of frame-annotation completeness.

## 2026-04-17T10:08:20.187Z | decision | high | Autosave physical touch and occlusion labels

Removed the manual save button from the physical transient-label flow and switched touch/occlusion edits to autosave. The client now persists changes automatically after each valid edit, serializes saves to avoid out-of-order overwrites, and suppresses infinite retry loops for unchanged failing payloads. UI keeps only a minimal `Saving…` indicator while autosave is in flight.

## 2026-04-17T14:50:44.912Z | plan | high | Audit physical reader families and projected-crop migration constraints

Repo audit found that the current production physical runtime is still the board-probe path in `pipeline/physical/square_classifier.py` (despite the module name), used by analysis/dev-tools/tracker eval. The projection-based crop work currently feeds only the two-stage occupancy/piece dataset path in `pipeline/physical/square_classifier_data.py`; its runtime `pipeline/physical/two_stage_board_reader.py` still uses legacy `square_crop.py`, so that lane has a train/runtime geometry mismatch. README is materially stale for the physical stack: it omits `pipeline/physical`, board-probe/direct-board/autoresearch workspaces, and does not explain which paths are active vs experimental. Cleanup planning should therefore distinguish whole-board readers from square-crop readers instead of assuming piece-projection crops can be dropped into every model family.

## 2026-04-17T15:05:51.713Z | decision | medium | Transient autosave waits for baseline sync before saving local edits

Investigated `/annotate/physical` autosave misbehavior on `clip_overlay_h2WrtkfwRl8_clip8_6.pt`. Root cause was `usePhysicalTransientAnnotations` firing autosave from intermediate draft state while saved annotation / corrected move baseline was still being hydrated, and then firing a duplicate save immediately after successful saves. Fix in `dev-tools/hooks/usePhysicalTransientAnnotations.ts`: track local-dirty state separately from raw draft-vs-baseline diffs, merge occlusion toggles against refreshed baselines, block autosave while a baseline-sync draft update is pending, and clear dirty state immediately on successful save acknowledgement.

## 2026-04-17T16:32:53.992Z | prompt | high | Physical stack reorganisation brief for refactor/piece_projection

User provided a detailed 12-phase refactor plan for `pipeline/physical/` with discovery gates before deletion. Key directives: tag `pre-cleanup-archive` on `main`, branch `refactor/piece_projection`, make `pipeline/physical/piece_projection.py` the sole geometry source, reorganize the physical stack into `board_probe/`, `two_stage/`, and `shared/`, delete the whole-board lane and all non-piece_projection geometry, promote the best autoresearch decoder config only after retrain + resweep, retrain surviving models, update README/progress docs, and keep `.agents/memory/*` out of the branch diff.

## 2026-04-17T16:42:33.053Z | decision | high | Discovery gated piece_projection refactor to deletion-safe scope

On branch `refactor/piece_projection`, Phase 0 was completed by tagging `pre-cleanup-archive` and branching from `main`. Phase 1 discovery found that board-probe geometry is currently split across rectified-board assumptions, oblique board crops, and heuristic oblique patch/square sampling. `pipeline/physical/piece_projection.py` already covers homography, camera pose, and projected piece/occupancy crops, but still needs reusable board-neighborhood crop and per-square bbox helpers while leaving patch→square pooling in the board-probe family. `autoresearch/prepare.py` caches runtime square logits with shape `(num_frames, 64, 13)`, so post-refactor logits can keep the current decoder surface if row-major `64 x 13` output is preserved. For the move-model/joint-reader entanglement, the chosen route is to remove oblique/native-oblique move-model modes and the joint-reader checkpoint bridge instead of preserving or generalizing them, which allows `pipeline/physical/joint_board_reader.py` and move-data oblique branches to be deleted cleanly in later phases.

## 2026-04-17T19:47:46.127Z | decision | high | Completed piece_projection physical-stack reorganisation

Finished the `refactor/piece_projection` reorganisation. The physical stack is now split into `pipeline/physical/board_probe/`, `pipeline/physical/two_stage/`, and `pipeline/physical/shared/` with `pipeline/physical/piece_projection.py` as the sole geometry module. Deleted the whole-board lane, oblique geometry lane, joint-reader bridge, legacy crop helpers, and dead scripts/tests. Board-probe runtime/training now run through the reorganized modules, autoresearch cache was regenerated against the refactor-era promoted checkpoint, the top decoder group was rechecked, and decoder `v282` was promoted as the production default via `pipeline/physical/board_probe/decoder.py` and `weights/physical/metadata.json`. Verification completed with full tests green, `make lint`, `make typecheck`, retrained/warm-start retrained board-probe and two-stage checkpoints meeting the required baselines, and local dev-tools smoke tests for the physical routes/endpoints.

## 2026-04-17T20:04:19.670Z | decision | medium | Physical annotation split-manifest writes made atomic

Debugged an intermittent `/annotate/physical/...?...split=val` internal server error. Root cause was `pipeline/physical/shared/splits.py` rewriting `data/physical/source_video_splits.json` with plain `write_text()`, which allowed concurrent reads during truncation and produced `JSONDecodeError: Expecting value` on the annotation API. Fixed by atomically replacing the manifest via a temp file + `replace()`, and by treating an existing blank manifest as recoverable empty state so the annotation route can rebuild it instead of 500ing.

## 2026-04-18T08:35:20.236Z | decision | high | Fix piece-projection review regressions after refactor

Implemented the review follow-up so the active square-based physical stack is geometrically consistent again.

- `pipeline/analysis/board_reading.py`: the segmentation fallback no longer runs the board-probe reader without corners; it falls back through VLM only.
- `pipeline/physical/board_probe/board_data.py`: annotated board datasets now read original clip frames from `clip_path`/`frame_index`, preprocess with stored corners, and skip legacy rows that lack clip metadata.
- `pipeline/physical/board_probe/runtime.py` + `probe.py`: projected square pooling now stays on-device, computes one homography per sample, and fast-paths full-frame boards via direct patch-grid pooling.
- `scripts/eval_physical_board_{runtime,tracker}.py`, `pipeline/physical/board_probe/{runtime_visualization,failure_study}.py`: eval/diagnostic tooling now loads clip frames and passes `row.corners`/`corners_list` instead of silently using rectified `row.board_path` boards.
- `pipeline/physical/shared/move_data.py` and `pipeline/physical/piece_projection.py`: board-neighborhood prep and occupancy bboxes now reuse shared helpers instead of duplicating geometry.
- `pipeline/physical/shared/source_video_paths.py`: source-video lookup now reuses `pipeline.paths.find_video_file()`.
- `scripts/train_physical_move_model.py`: removed the dead `--initialize-square-reader-checkpoint` flag.
- Weight summaries now point at committed checkpoint paths under `weights/physical/...`.

Validation: `make lint`, `make typecheck`, `make test`.

## 2026-04-18T08:38:01.495Z | decision | low | Physical runtime inspector still labels board images as rectified despite new piece-projection ru...

Confirmed `/evaluate/physical` runtime inspector sessions (e.g. `22725862d6ce`) are using the promoted default model `weights/physical/best.pt` / `default · v8r1`, whose metadata says `board_input_mode: piece_projection_board`. Runtime inference goes through `pipeline/physical/board_probe/runtime.py` (`preprocess_board_neighborhood_image` + `sample_projected_square_tokens_from_patch_tokens`) and loads frames from `clip_path`, not from `rectified_board_path`. The dev-tools UI copy is stale (`dev-tools/components/evaluate/PhysicalRuntimeCard.tsx` says `Rectified frame crop`; `PhysicalRuntimeInspector.tsx` tooltip says `Samples rectified...`), and `PhysicalEvalBoardRow.board_path` is still populated from legacy `rectified_board_path` metadata for display/debug only.

## 2026-04-18T08:54:51.502Z | decision | low | Physical runtime inspector now shows board-probe geometry and hoverable square evidence

Updated `/evaluate/physical` runtime cards so the image panel now presents the oblique board-neighborhood input geometry used by the board-probe runtime instead of labeling it as rectified. Backend runtime inspection now emits normalized projected square quads (`geometry_square_quads`) derived from the same geometry preview. Frontend overlays the projected grid, rehydrates older sessions missing this metadata, and hovering any GT/stateless/temporal board square shows the corresponding projected evidence region used for token pooling.

## 2026-04-18T09:46:09.047Z | experiment | high | Post-fix retraining comparison across board-probe, two-stage, and move-model paths

Ran the three physical-reader/tracker training paths sequentially after fixing the review regressions, then re-evaluated on val.

Artifacts:
- board-probe: `outputs/2026-04-17/compare_afterfix_board_probe_pieceproj_manualonly_init_rectified_epoch5/metrics.json`
- two-stage: `outputs/2026-04-17/compare_afterfix_square_classifier_{occupancy,piece}/summary.json` and `outputs/2026-04-17/compare_afterfix_two_stage_eval.json`
- direct move model: `outputs/2026-04-17/compare_afterfix_move_model_realonly_allframes_w01/summary.json` and `outputs/2026-04-17/compare_afterfix_move_model_eval_moveprob_margin_0.5.json`
- comparison report: `outputs/2026-04-17/compare_afterfix_physical_paths.md`

Headline results vs latest comparable prior runs:
- Board-probe piece-projection rerun regressed sharply once train/eval switched to the corrected clip-frame + corners contract: square accuracy `0.6085 -> 0.4705`, non-empty accuracy `0.3233 -> 0.1746`, macro F1 `0.2886 -> 0.1212`, board exact stayed `0.0`. This indicates the prior promoted piece-projection numbers were inflated by the pre-fix rectified-board leakage/mismatch.
- Two-stage warm-start retrain was effectively unchanged: per-square accuracy `0.7669 -> 0.7683`, empty `0.8630 -> 0.8663`, non-empty `0.5944 -> 0.5925`, occupancy `0.8333 -> 0.8344`, board exact stayed `0.00355`.
- Direct move-model rerun (same real-only all-frames/no-move-weight-0.1 recipe, `move_prob_margin=0.5` eval) changed from freezing to overfiring: move recall `0.0 -> 0.9375`, false-change rate `0.0 -> 0.8810`, while square accuracy `0.8736 -> 0.6788`, non-empty `0.8353 -> 0.3641`, macro F1 `0.7683 -> 0.3811`, board exact `0.0604 -> 0.0`.

Interpretation:
- After the geometry/data-contract fix, two-stage is the only path whose held-out behavior remained stable.
- Corrected piece-projection board-probe and direct move-model paths both need renewed search/tuning because their latest pre-fix metrics were not trustworthy under the now-correct input contract.

## 2026-04-18T12:32:40.323Z | experiment | medium | Apples-to-apples production-decoder eval on corrected post-fix board-probe checkpoint

Ran `scripts/eval_physical_board_tracker.py` with `tracker-mode=production` (`v282`) against the corrected retrained board-probe checkpoint `outputs/2026-04-17/compare_afterfix_board_probe_pieceproj_manualonly_init_rectified_epoch5/board_probe.pt` and saved `outputs/2026-04-17/tracker_eval_compare_afterfix_board_probe_production.json`.

Result vs old promoted production checkpoint:
- old promoted checkpoint (`outputs/2026-04-17/tracker_eval_weights_best_production.json`): `board_exact=0.2544`, `accuracy=0.9120`, `non_empty=0.8472`, `macro_f1=0.8443`, `move_recall=0.4844`, `false_change_rate=0.0155` (`12/773` static false-change frames, `31/64` matched GT changes)
- corrected post-fix retrain: `board_exact=0.0355`, `accuracy=0.8567`, `non_empty=0.7928`, `macro_f1=0.7599`, `move_recall=0.4063`, `false_change_rate=0.0362` (`28/773` false-change frames, `26/64` matched GT changes)

Implication: the huge gap to `autoresearch/` is only partly an apples-to-oranges issue. Using the same production decoder lifts the corrected retrain far above its raw per-frame `board_exact=0.0`, but the main regression remains in the underlying logits and the decoder was tuned on the older promoted checkpoint, not the corrected retrain.

## 2026-04-18T12:35:52.359Z | decision | medium | Board-probe piece_projection_board still uses flat square pooling, not piece-box projection

Inspection after the runtime-inspector geometry UI change showed that the promoted board-probe path (`pipeline/physical/board_probe/probe.py::sample_projected_square_tokens_from_patch_tokens`) computes pooling regions from planar board-surface square bboxes via `_project_square_bboxes_from_corners`, not from 3D piece-box projection. The expected review visualization (`outputs/piece_projection_review2/*`) comes from `pipeline/physical/piece_projection.py::project_piece_box` / `extract_projected_piece_crop`, which is a different geometry path. So the runtime inspector's square hover regions were faithful to current board-probe code, but current `piece_projection_board` semantics do not match the per-piece projection review outputs.

## 2026-04-18T12:40:12.565Z | decision | high | Autoresearch 25% board-exact result was on stale rectified-logit surface, not corrected piece-pro...

Investigated the gap between the `autoresearch/` board-exact numbers (~`0.2544`) and the corrected post-fix runtime results.

Confirmed facts:
- `autoresearch/prepare.py` still builds `autoresearch/cache/rectified_realplusmanual_board_logits.pt` by loading `row.board_path` via `_load_rectified_board_image(...)`, i.e. rectified board images, not the corrected clip-frame + corners piece-projection input contract.
- Rerunning the *current* tracker eval (`scripts/eval_physical_board_tracker.py`, which now loads clip frames via `load_annotated_board_frame_bgr`) on `weights/physical/best.pt` no longer reproduces `0.2544`; it yields `board_exact=0.0355`, `accuracy=0.8567`, `non_empty=0.7928`, `macro_f1=0.7599` (`outputs/2026-04-18/tracker_eval_weights_best_production_rerun.json`).
- The post-fix retrain artifact `outputs/2026-04-17/compare_afterfix_board_probe_pieceproj_manualonly_init_rectified_epoch5/board_probe.pt` has a state_dict exactly identical to `weights/physical/best.pt` / `outputs/2026-04-17/physical_board_probe_pieceproj_manualonly_init_oldcheckpoint_epoch5_posmlp_promoted/board_probe.pt`. The 5-epoch retrain did not beat the initialized checkpoint on its synthetic selection set, so it preserved the init weights.

Implication:
- The apparent drop from `25%` board exact to `~3.5%` is primarily an evaluation/input-contract correction, not evidence that the newer software architecture trained a genuinely worse checkpoint.
- Autoresearch currently optimizes decoder behavior on a stale rectified-logit cache and must be rebuilt against the corrected clip-frame + corners runtime before its wins can be treated as valid for the active physical stack.
- The current piece-projection retrain recipe also needs a better selection surface (likely real clip-frame/corners, not only synthetic val) because it can trivially freeze on the initialization checkpoint.

## 2026-04-18T12:54:40.994Z | decision | medium | Board-probe piece_projection_board migrated to projected piece-box pooling

Changed `pipeline/physical/board_probe/probe.py::sample_projected_square_tokens_from_patch_tokens` so board-probe square-token pooling now uses projected 3D piece-box bboxes from `pipeline/physical/piece_projection.py::project_piece_bboxes`, not flat board-surface square bboxes. Updated runtime visualization / physical runtime inspector to expose both base board quads and projected piece bboxes; hover preview now shows the projected piece-box crop instead of a planar square region. Existing promoted weights (`weights/physical/best.pt`) now run under the new pooling semantics and should be re-evaluated/retrained before treating current metrics as comparable to the old surface.

## 2026-04-18T13:12:53.556Z | experiment | medium | Piece-box pooling retrain sweeps did not yield a promotable board-probe checkpoint yet

Ran CPU sweeps after migrating board-probe pooling to projected piece-box bboxes.

Artifacts:
- `outputs/2026-04-18/runtime_eval_piecebox_best_off.json` — current promoted `weights/physical/best.pt` under new pooling (`accuracy 0.5337`, `non_empty 0.0529`, `macro_f1 0.1054`)
- `outputs/2026-04-18/piecebox_manual_selection_sweep/summary.json` — manual-only retrain with manual selection holdout `EEZo0uDh4AY`; best tradeoff was `scratch.pt` but still poor (`accuracy 0.1592`, `non_empty 0.2050`, `macro_f1 0.1062`)
- `outputs/2026-04-18/piecebox_mixed_manual_selection_sweep/summary.json` — synthetic+manual retrain with manual selection holdout; best tradeoff was `init_pieceproj_seed1.pt` (`accuracy 0.2218`, `non_empty 0.2159`, `macro_f1 0.1280`)

Conclusion: the piece-box pooling migration is implemented and the inspector now visualizes the intended tall piece regions, but no retrained board-probe checkpoint from these quick sweeps is strong enough to promote over the existing default. Promotion was intentionally withheld pending a better training/selection recipe (likely requiring a more suitable real selection surface and/or architecture changes).

## 2026-04-18T14:17:33.819Z | decision | high | Physical failure-study contact sheet is not rendering active piece-projection geometry

Found that `outputs/2026-04-18/physical_board_failure_study_weights_best_corrected/contact_sheet.png` is a misleading visualization for the current physical stack. `pipeline/physical/board_probe/failure_study.py::_render_failure_frame` passes the full clip frame from `load_annotated_board_frame_bgr(...)` into `_render_crop_panel`, and `_render_crop_panel` simply resizes that image and draws a naive 8x8 grid. It does **not** render the active board-neighborhood crop or projected piece-box pooling geometry.

This is inconsistent with the actual training/runtime contract, which still uses:
- `board_data.py::preprocess_board_neighborhood_image(...)` -> `piece_projection.extract_board_neighborhood_crop(...)`
- `runtime.py::_prepare_runtime_board_input(...)`
- `probe.py::sample_projected_square_tokens_from_patch_tokens(...)` with `project_piece_bboxes(...)`

Implication: the corrected-runtime contact sheet should not be used to reason about current geometry quality or failure mix until the renderer is updated to match the same piece-projection visualization used by `outputs/piece_projection_review2/` and the runtime inspector.

## 2026-04-18T14:17:51.211Z | experiment | medium | Generated visual board-probe pipeline walkthrough artifact

Created a reusable diagnostic script `scripts/visualize_physical_board_probe_pipeline.py` and generated a user-facing walkthrough under `.agents/memory/attachments/physical_board_probe_pipeline_walkthrough/`.

Artifact contents:
- `summary.md`
- `01_real_input.png`
- `02_real_geometry_overlay.png`
- `03_real_patch_usage_heatmap.png`
- `04_real_square_examples.png`
- `05_real_probe_boards.png`
- `06_real_vs_synthetic_pooling.png`
- `stats.json`

Key takeaway surfaced by the walkthrough: the piece-box migration changed tokenization, not just UI crops. On the inspected real oblique frame, projected piece-box pooling used 6–20 patches per square (mean 12.05), with 124/256 patches shared by multiple squares and adjacent square-token cosine rising from 0.734 (planar) to 0.944 (piece-box). The synthetic comparison showed much weaker overlap (`avg_squares_per_used_patch 2.25` synthetic vs `5.43` real), highlighting a train/eval geometry gap worth investigating.

## 2026-04-18T14:23:34.292Z | decision | medium | Failure-study geometry panel now renders board-neighborhood crop with projected piece boxes

Updated `pipeline/physical/board_probe/failure_study.py` so failure-study episode frames and contact sheets no longer overlay a naive 8x8 grid on the full source frame. The first panel now renders the active piece-projection geometry contract:

- board-neighborhood crop from `extract_board_neighborhood_crop(...)`
- projected square quads from `project_square_base_quad(...)`
- projected piece pooling boxes from `project_piece_bboxes(...)`
- GT/predicted change highlights preserved on top of that geometry

Rebuilt `outputs/2026-04-18/physical_board_failure_study_weights_best_corrected/contact_sheet.png`, which now visually matches the piece-projection review semantics instead of the old misleading visualization. Validation passed: `make lint`, `make typecheck`, `make test`.

## 2026-04-18T14:40:47.917Z | experiment | medium | Added correct two-stage training pipeline walkthrough artifact for cQAedm_gWrw frame

Created `.agents/memory/attachments/two_stage_training_pipeline_cQAedm_gWrw_frame0000/` using `scripts/visualize_two_stage_training_pipeline.py` to show the actual crop-based two-stage training path on `clip_overlay_cQAedm_gWrw_clip67_2_frame0000`.

Contents:
- `01_input_sources.png` — confirms the two-stage datasets use the native source-video frame (`1920x1080`) rather than the stored `224x224` clip frame when native metadata exists
- `02_geometry_overlay.png` — projected square quads, occupancy bboxes, and occupied-piece bboxes
- `03_occupancy_contact_sheet.png` — all 64 occupancy samples (`112x112`)
- `04_piece_contact_sheet.png` — 28 occupied piece samples (`224x224`), matching the crop-based geometry path the user referenced from `outputs/piece_projection_review2/*`
- `05_selected_square_examples.png` — step-by-step examples for `c4`, `b4`, and `g5`
- `summary.md` and `stats.json`

Key numbers for this frame:
- occupancy samples: 64
- piece samples: 28
- source kind: native frame
- occupancy bbox size mean: `120.5x54.3 px`
- occupied piece bbox size mean: `85.9x148.5 px`

Important surfaced details:
- the two-stage path is truly crop-based, not whole-board token pooling
- occupancy and piece branches consume different geometry
- left-half piece crops (`a`-`d`) are horizontally flipped before training
- empty squares train occupancy only and are skipped by the piece classifier

## 2026-04-18T14:40:59.923Z | decision | high | Board-probe input contract bug: active path was still using degraded 224 clip frames instead of n...

Confirmed the user's geometry complaint. `outputs/piece_projection_review2/` is generated by `scripts/visualize_piece_projections.py`, which uses native source-video frames plus `native_image_bbox`/`native_corners` to build full-frame corners before projection. By contrast, the active board-probe data path in `pipeline/physical/board_probe/board_data.py` had still been loading degraded `clip_path` tensors (`224x224`) and clip-frame corners, because `load_annotated_board_rows()` discarded native metadata and `load_annotated_board_frame_bgr()` always read the clip tensor.

Fixes made:
- `board_data.py` now preserves native annotation metadata (`source_frame_index`, `native_corners`, `native_image_bbox`, `clip_frame_size`) and prefers native source-video frames when available.
- `row.corners` now resolve to full-frame native corners when native metadata exists.
- Added tests covering native-corner parsing and native-frame loader preference.
- `failure_study.py` geometry panel now renders the exact piece-projection overlay signature (full frame + projected piece boxes/base quads, letterboxed to panel) instead of the old naive grid/crop view.
- Rebuilt `outputs/2026-04-18/physical_board_failure_study_weights_best_corrected/contact_sheet.png`.

Impact check:
- Rerunning `scripts/eval_physical_board_tracker.py` on `weights/physical/best.pt` with the native-frame board-probe path produced `outputs/2026-04-18/tracker_eval_weights_best_production_native.json` with `board_exact=0.05917`, improving over the prior clip-frame rerun (`0.03550`) but still far below the stale rectified/autoresearch surface (`0.25444`).

Validation passed: `make lint`, `make typecheck`, `make test`.

## 2026-04-18T20:25:52.287Z | experiment | medium | Ran inspectable two-stage subset training with exported on-the-fly crops

Created a CPU-feasible two-stage inspection run under `data/inspection/two_stage_training_run_2026-04-18_subset128x128_noaugment/`.

What was done:
- sampled `128` train + `128` val boards into `subset/{train,val}/board_annotations.jsonl`
- exported the exact deterministic no-augment crops used by the run into `dataset_images/` for both occupancy and piece tasks
- ran `scripts/train_square_classifier.py` for both tasks on that subset

Results:
- occupancy (`112x112`, batch 64, 4 epochs): best val accuracy `0.68212890625`
- piece (`224x224`, batch 32, 4 epochs): best val accuracy `0.41168384879725084`

Artifacts:
- run summary: `data/inspection/two_stage_training_run_2026-04-18_subset128x128_noaugment/run_summary.md`
- occupancy log: `.../training_occupancy.log`
- piece log: `.../training_piece.log`
- exact run crops: `.../dataset_images/{train,val}/{occupancy,piece}/images/`

Also exported the full deterministic no-augment two-stage crop corpus (without completing a full-data training run) to `data/inspection/two_stage_training_run_2026-04-18_default_noaugment/dataset_images/`. The attempted full-data occupancy training run was not completed because this environment is CPU-only and the interactive timeout was too tight for the full held-out split.

## 2026-04-18T22:37:33.892Z | decision | high | Physical runtime viewer mismatch is a crop-preview artifact; move_data still has clip-frame corne...

Investigated `/evaluate/physical/<session>` against `outputs/piece_projection_review2/*`. The runtime card is not rendering the full-frame source-of-truth overlay: `physical_runtime_service._runtime_result_from_frame()` stores only `thumbnail_b64 = frame.crop_rgb`, `runtime_visualization._runtime_geometry_preview()` first applies `extract_board_neighborhood_crop()` and rescales to a square preview, and `PhysicalRuntimeCard` draws SVG quads/bboxes over that crop. `failure_study.py` / `scripts/visualize_piece_projections.py` instead render full-frame projected overlays, so the viewer can look 'wrong' even when geometry is the same crop-transformed signature. Important footgun found repo-wide: `pipeline/physical/shared/annotation_rows.py` exposes raw `corners` in `corner_space=clip_frame`, but some consumers still ignore that and use `row.corners` + `_load_clip_frame_bgr()` directly. The concrete training/eval-adjacent offender is `pipeline/physical/shared/move_data.py::load_eval_move_sequences()`, used by physical move-model eval / decoded selection, which still builds projected inputs from degraded clip frames and clip-space corners instead of native source frames + resolved full-frame corners. Board-probe runtime/training/autoresearch paths are on the corrected native-frame contract via `pipeline/physical/board_probe/board_data.py` and `load_annotated_board_frame_bgr()`.

## 2026-04-18T22:39:31.107Z | decision | high | Autoresearch prepare now uses native-frame board-probe contract

Updated `autoresearch/prepare.py` to stop caching logits from rectified `row.board_path` boards. It now rebuilds `autoresearch/cache/native_realplusmanual_board_logits.pt` from `load_annotated_board_frame_bgr(...)` plus `row.corners`, defaults its baseline report to `outputs/2026-04-18/tracker_eval_weights_best_production_native.json`, and verifies by replaying the baseline report's tracker mode instead of assuming lookahead. Added tests in `tests/test_autoresearch_prepare.py` and reran `.venv/bin/python3 autoresearch/prepare.py --force`, which reproduced `board_exact=0.05917159763313609` on the corrected native surface.

## 2026-04-18T22:45:48.188Z | experiment | low | Added prepare-script example attachments

Created user-facing attachments under `.agents/memory/attachments/` to explain `autoresearch/prepare.py` on one cached board: `prepare_example_full_frame.png` (native frame with board corners), `prepare_example_board_crop_224.png` (224x224 board neighborhood view passed into runtime after crop+resize), and `prepare_example_summary.json` (annotation id, clip/frame metadata, corners, crop size, logits shape, and a small prediction summary).

## 2026-04-18T22:58:39.048Z | decision | high | Physical runtime page now uses full-frame projection overlays; annotated move-data eval uses nati...

Implemented the evaluation-page fix and the remaining annotated move-data footgun fix.

- `pipeline/physical/board_probe/runtime_visualization.py` now builds two geometries per selected frame: the existing board-neighborhood crop preview for thumbnails/contact sheets, and source-frame-normalized full-frame geometry for evaluation. It also renders a full-frame piece-projection overlay image matching the `piece_projection_review2` / failure-study contract.
- `dev-tools/api/services/evaluate/physical_runtime_service.py` now returns `image_b64` as that full-frame overlay plus `geometry_space = source_frame_normalized`, while keeping the crop thumbnail for session chips.
- `dev-tools/components/evaluate/PhysicalRuntimeCard.tsx` now prefers the full-frame overlay image, uses a wider runtime-input panel, and keeps hover-driven piece-box crop inspection from the returned geometry.
- `dev-tools/components/evaluate/PhysicalRuntimeInspector.tsx` now rehydrates legacy sessions when geometry/image payloads are missing or still on the old crop-space contract.
- `pipeline/physical/shared/move_data.py::load_eval_move_sequences()` now uses `load_annotated_board_rows()` + `load_annotated_board_frame_bgr()`, so annotated move-model eval/selection no longer silently uses 224x224 clip frames plus `corner_space=clip_frame` coordinates.
- Verified with targeted tests plus `make typecheck`, `make lint`, and `make test`.

Direct inspection of `clip_overlay_e4lGbQp4pU4_clip70_9_frame0073` now produces a full-frame overlay matching the expected `outputs/piece_projection_review2` geometry signature instead of the old square-cropped preview artifact.

## 2026-04-18T23:07:16.924Z | experiment | medium | Screening regression comes from new-channel/domain shift and non-DINO heuristics, not changed wei...

Investigated `/evaluate/screening` against current `weights/screening` checkpoint (`v3r7`, trained 2026-04-04). The classifier path is still the same frozen DINO encoder + MLP head, but predictions also depend on heuristic overlay/OTB scores in `pipeline/screen/ai_classifier.py` and a hard vertical auto-reject in `pipeline/screen/frame_fetcher.py` / `dev-tools/api/services/evaluate/models_service.py`.

Findings from local inspection:
- Current labeled pool is dominated by AI-written labels (`screened_by='ai'`: 62792 / 68133), so random evaluation over all screened videos is heavily self-referential and can hide real regressions.
- On the recent post-`v3r7` human-reviewed slice checked locally (110 videos), current model agreement was only 52/110 (~47%).
- 11 of 58 mismatches were false vertical detections that forced `reject` with confidence 1.0 on landscape/dark-edge broadcast footage.
- Remaining failures cluster on newer channel styles that were weakly represented or absent in the earlier training surface, especially `@FIDEchess`, `@chess24`, `@crestbook`, and `@AnnaCramling`.
- Example failure modes: OTB-only broadcasts misread as `overlay` because heuristic overlay scores fire on score/lower-third graphics; wide hall/preshow shots labeled `reject` but predicted `overlay`; dark-background broadcasts falsely auto-rejected as `vertical`.

Conclusion: the apparent screening drop is driven by data/label/domain shift plus brittle heuristics outside the frozen DINO head, not by a recent DINO or checkpoint change.

## 2026-04-18T23:15:52.420Z | decision | high | Physical runtime page switched from pre-rendered overlay image to raw-frame client overlay

Adjusted the just-landed `/evaluate/physical` fix after user feedback. The page no longer shows a server-rendered full-frame overlay image as its main panel. Instead, runtime inspection returns the raw source frame (`image_render_mode = source_frame_raw`) plus source-frame-normalized quads/bboxes, and `PhysicalRuntimeCard` draws the yellow projections client-side over the original image. Hovering a board square now highlights that square’s projected geometry in fuchsia on the original image while keeping the projected piece-box crop preview below. `PhysicalRuntimeInspector` now rehydrates legacy sessions if they still carry the older pre-rendered-image payload or other stale geometry contract state.

## 2026-04-18T23:59:09.772Z | decision | high | Board-probe pooling now maps full-frame piece boxes into crop space before token pooling

Fixed the remaining geometry leak after the native-frame migration. `pipeline/physical/board_probe/*` no longer reprojects piece boxes from resized crop-space corners for real oblique boards. It now projects piece boxes in full-frame source geometry, then maps those bboxes through the board-neighborhood crop+resize transform before pooling DINO patches. Regenerated the matching baseline report at `outputs/2026-04-18/tracker_eval_weights_best_production_native_pieceboxmapped_off.json` and updated `autoresearch/prepare.py` to verify against that surface; `.venv/bin/python3 autoresearch/prepare.py --force` now passes again.

## 2026-04-19T00:18:03.779Z | decision | medium | Fixed authresearch typo across autoresearch workspace

Renamed stale `authresearch` references across the local `autoresearch/` workspace to `autoresearch`, including active controllers/docs (`controller.sh`, `controller_slow.sh`, `program.md`), current scripts (`prepare.py`, `train.py`, `best_train.py`), pyproject metadata, and historical workspace artifacts/logs under `autoresearch/`. Controller env vars are now `AUTORESEARCH_*`, and `autoresearch/controller.sh` now points directly at `autoresearch/train.py` without needing a temporary `authresearch` symlink.

## 2026-04-19T01:13:49.082Z | experiment | high | MPS reruns and corrected autoresearch cache on native piecebox surface

- Verified PyTorch MPS now works in `.venv`; reran physical training/eval on MPS.
- Two-stage 10-epoch MPS retrains landed at `outputs/2026-04-18/two_stage_native_occupancy_10ep_mps/summary.json` (`best_val_accuracy=0.8372226331`) and `outputs/2026-04-19/two_stage_native_piece_10ep_mps/summary.json` (`best_val_accuracy=0.7343806522`); eval `outputs/2026-04-19/two_stage_eval_native_10ep_mps.json` stayed effectively flat (`per-square=0.7763`, `non_empty=0.5711`, `board_exact=0.0036`).
- Whole-board 10-epoch MPS retrain `outputs/2026-04-19/physical_board_probe_pieceproj_manualonly_init_bestpt_10ep_mps/metrics.json` regressed on raw board metrics (`real_eval accuracy=0.5201`, `non_empty=0.1734`, `macro_f1=0.1340`, `board_exact=0.0`); production tracker eval `outputs/2026-04-19/tracker_eval_board_probe_10ep_mps_production.json` matched the weak native baseline surface (`board_exact=0.0378698225`).
- `autoresearch/prepare.py --device mps --force` successfully rebuilt `autoresearch/cache/native_realplusmanual_board_logits.pt` and verified the corrected native piecebox baseline against `outputs/2026-04-18/tracker_eval_weights_best_production_native_pieceboxmapped_off.json`.
- Current autoresearch plateau on the corrected cache remains `board_exact=0.0745562130`; probes lowering gates, pure segmental decode, extra dropped worst frames, and stricter board-change threshold all regressed or were no-ops.

## 2026-04-19T01:14:26.081Z | decision | high | Autoresearch best-run bookkeeping now compares against exact best_result.json

`autoresearch/prepare.py:record_successful_run()` no longer decides keep/discard from rounded `results.tsv` entries or the old same-config auto-keep path. It now compares the candidate run against the exact current best stored in `autoresearch/best_result.json` (falling back to the log only if needed), and a unit test in `tests/test_autoresearch_prepare.py` covers the regression where a worse rerun of the current config could overwrite the best because the TSV had rounded away precision.

## 2026-04-19T07:44:12.719Z | decision | medium | Physical transient annotation move selection and autosave fix

Updated `/annotate/physical` transient labeling UX so move-row selection no longer changes the current frame, enabling reuse of one frame as another move's touch boundary. Added `i/o/p` shortcuts for `touchstart`, `touchend`, and hand occlusion with inline keycaps. Reworked `usePhysicalTransientAnnotations` autosave to save directly from ref-backed draft snapshots on each user mutation, preserve later edits during in-flight saves, and clear stale baseline-sync blockers that were preventing immediate saves.

## 2026-04-19T07:57:40.803Z | prompt | high | User wants side-by-side comparison against Chesscog occupancy/piece classifiers

User wants to compare Argus occupancy classification and piece classification against the Chesscog implementation/report (`~/dev/chesscog`, `~/dev/chesscog/docs/report.pdf`) using Argus data, with a side-by-side evaluation of both approaches.
