# Memory log

## 2026-04-16T18:59:43.849Z | experiment | low | Added physical crop types comparison attachment

Created `.agents/memory/attachments/physical_crop_types_example.png`, a user-facing 6-panel visual showing one held-out physical example through: source frame, native board-neighborhood crop, oblique board crop, rectified board, rectified 64-tile split, and oblique square-context crops. Useful for explaining the repo's crop families and how rectified vs oblique representations differ.

## 2026-04-16T19:25:49.017Z | prompt | high | Diagnosis-first plan for pushing board_exact toward 90%

User pointed to `outputs/plan.md`, which sets the objective to push `board_exact` as close to 90% as the current data and architecture support on the evaluated target split. Key constraints: state current baseline before changes; first run a failure study on the best rectified+lookahead run; sample 100 failure episodes (first failing frame + ~10 preceding frames, capped per video); tag failures into rectification/localization, piece-classifier/square-evidence, temporal ambiguity, decoder/legal-hypothesis/error-propagation, eval/label issue, or other; build a viewer to scrub leading frames for each tagged failure; choose subsequent experiments only based on those buckets; avoid architecture sweeps without failure-mode justification; abandon low-return directions; use oracle if experiment results contradict hypotheses or when failure data does not disambiguate.
