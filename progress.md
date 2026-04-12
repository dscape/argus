# Progress

## 2026-04-12

### Physical annotation UI
- Kept the frame navigation controls in `dev-tools/components/annotate/PhysicalAnnotationPage.tsx` at fixed widths so the controls stay in the same place while reviewing.
- Changed the physical-board action area to show explicit per-frame save state (`Saved` vs `Save needed`).
- Replaced the small save action with a primary `Save + Next >` flow so a successful save advances one frame for faster review.
- Save success now relies on the persistent status pill and frame advance instead of firing a success toast on every frame.
- Fixed FEN prefilling on move frames so the editable board now uses the same post-move replay position shown in the computed replay panel.
- Added clip-level completion state to the physical annotation index so fully annotated clips show a checkmark and saved-frame progress.
- Made the move list under computed replay expand to fill the available center-column height so long games are easier to review.
- Added manual move correction: saved annotations now produce corrected replay/move data on the physical annotation page, including newly inferred or replaced moves from full-board or partial square edits.
- Replaced the unused `FEN` action with `Copy FEN from previous` so reviewers can quickly carry forward the prior frame's board state while correcting sequences.
- Added `Copy FEN from next` for cases where the visible board change lands a frame late and the next frame is the better correction source.
- Tightened the physical annotation action-bar button styling and shortened the copy labels to keep the toolbar from overflowing and shifting.
- Added draggable corner handles on the source frame with debounced live re-rectification plus a final rectify on drag release, so moving-camera clips can be adjusted frame by frame without re-clicking all four corners.

### Held-out physical eval set
- Manual annotation produced a held-out physical-board eval set under `data/physical/eval/`.
- Current summary from `/api/physical-eval/summary`:
  - `846` annotated board frames
  - `54,142` labeled square crops
  - `4` held-out source videos
- Added structural training exclusion for those held-out videos:
  - new command: `python -m pipeline physical-split-clips`
  - this excludes source video ids found in `data/physical/eval/board_annotations.jsonl` from the exported physical train/val split.

### Validation
- Passed: `make typecheck`
- Passed: `make lint`
- Passed: `make test`
