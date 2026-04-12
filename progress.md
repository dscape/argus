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

### Physical square-classification baseline
- Added dedicated physical square-classification code under:
  - `pipeline/physical/square_data.py`
  - `pipeline/physical/square_probe.py`
  - `pipeline/physical/square_classifier.py`
  - `scripts/train_physical_square_classifier.py`
- Added held-out exclusion-aware training split support via:
  - `pipeline/physical/training_dataset.py`
  - `python -m pipeline physical-split-clips`
- Baseline report outputs:
  - `outputs/2026-04-12/physical_square_probe_smoke/`
  - `outputs/2026-04-12/physical_square_probe_baseline/`
- First frozen-DINO square-crop linear-probe result (synthetic train -> held-out real eval subset):
  - synthetic val accuracy: `0.5077`
  - held-out real square accuracy: `0.0589`
  - held-out real non-empty accuracy: `0.0534`
  - held-out real macro F1: `0.0378`
  - held-out real board exact match: `0.0176`
- Conclusion:
  - a frozen DINOv2 linear probe on independent physical square crops is not remotely good enough
  - the next physical per-square attempt should use **board context** rather than treating every square independently
- Added a board-context frozen DINO probe baseline under:
  - `pipeline/physical/board_data.py`
  - `pipeline/physical/board_probe.py`
  - `scripts/train_physical_board_probe.py`
- Board-context results:
  - top-down synthetic board renders improved transfer over independent square crops but are still bad:
    - `outputs/2026-04-12/physical_board_probe_large/`
    - held-out real square accuracy: `0.2451`
    - held-out real non-empty accuracy: `0.2355`
    - held-out real macro F1: `0.0773`
  - perspective-rendered synthetic boards from `argus.datagen.synth` were an even worse match when used directly, because the real eval boards are already **rectified** while those synthetic images are not.
- Updated assessment:
  - the next per-square step is still within the original objective, but it needs a better synthetic source: **rectified physical-board renders with realistic oblique piece appearance**, not top-down boards and not unrectified camera renders.

### Validation
- Passed: `make typecheck`
- Passed: `make lint`
- Passed: `make test`
