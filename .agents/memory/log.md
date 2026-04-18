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
