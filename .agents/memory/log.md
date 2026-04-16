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
