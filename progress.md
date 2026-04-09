# Progress

## Current architecture findings

- `pipeline/` is the real-data ingestion, labeling, calibration, and evaluation stack. It is not the final Argus runtime model.
- The implemented parts in `pipeline/` are meaningful and already useful:
  - crawl / screen / download videos
  - detect overlays and OTB regions
  - calibrate overlay and camera crops
  - read board state from overlays
  - detect moves from overlay FEN sequences
  - generate `.pt` training clips from camera footage
  - generate `.pt` training clips from synthetic footage generated with blender
- `pipeline/overlay/overlay_clip_generator.py` is the key bridge into model training. It already emits training tensors in the format expected by `src/argus/data/dataset.py`.

## What exists in `src/argus/`

- The chess core is implemented and solid:
  - move vocabulary
  - legality masks
  - game state machine
  - PGN generation
- The crop-level model path is implemented:
  - DINOv2 vision encoder
  - temporal model (`MambaTemporalModule` with GRU fallback)
  - constrained move head
  - trainer / evaluator
  - single-board inference from a sequence of board crops
- This means the repo already has a real **board-crop sequence -> legal move sequence -> PGN** path.

## What is not implemented end-to-end yet

- The real target system is **full camera input -> multi-board tracking -> per-board PGNs**.
- Augmenting training with synthetic data
- That full path is only partially scaffolded today.
- Existing scaffolding:
  - `src/argus/model/board_detector.py`
  - `src/argus/model/board_id_head.py`
  - `src/argus/data/collate.py` multi-board collation
  - `src/argus/inference/tracker.py`
  - phase2 / phase3 configs
- Critical missing pieces:
  - ROI pooling / crop extraction from predicted board boxes in `ArgusModel.forward_multi_board()`
  - multi-board inference loop in `src/argus/inference/pipeline.py`
  - training path that actually runs with `use_detector=True`
  - a real detector/identity training dataset wired into the trainer

## Architectural implication

- `pipeline/` should stay the **data factory** and debugging/evaluation surface for real videos.
- `src/argus/` should own the actual **camera-input multi-board model**.
- The overlay pipeline should be treated as training-data infrastructure and supervision tooling, not as the final Argus runtime architecture.



## End-to-end validation notes
- Training-clip generation blockers found during validation and fixed on this branch:
  - `pipeline.cli generate-clips` still called `get_video_path()` with the old two-arg signature.
  - generated clip filenames (`overlay_*.pt`) did not match `ArgusDataset`'s `clip_*.pt` glob.
  - generated `.pt` clips were missing PGN / timing metadata.
  - delay compensation mixed sampled-frame steps with raw frame indices.
  - clip replay always started from the standard position, so mid-game segments produced invalid legal masks.
  - invalid replay segments are now dropped instead of silently writing illegal move targets.
- Real-video clip generation validated end-to-end on five local videos under 200MB, with `.pt` outputs that now include PGN, timestamps, and frame tensors:
  - added an `@AnnaCramling` calibration to `configs/annotate/overlay_layouts.yaml` via `auto-calibrate --apply` so a fifth legal clip could be produced from local data.
  - `2wWUKmCBr6A` → `data/argus/training_clips/clip_overlay_2wWUKmCBr6A_clip5_0.pt`
  - `7RaBQag34Hk` → `data/argus/training_clips/clip_overlay_7RaBQag34Hk_clip26_0.pt`
  - `vkoTN5DxRS0` → `data/argus/training_clips/clip_overlay_vkoTN5DxRS0_clip22_1.pt`
  - `9h4IE1G99OE` → `data/argus/training_clips/clip_overlay_9h4IE1G99OE_clip69_1.pt`
  - `YEjQAF0hbBs` → `data/argus/training_clips/clip_overlay_YEjQAF0hbBs_1.pt` (generated with `--min-moves 3` to allow a short but legal segment)
  - all five inspect cleanly via `python -m pipeline.cli inspect-clip --file ...` and replay as legal games from their stored `initial_board_fen`.
- Inference validation is currently blocked by model quality, not tooling:
  - `make eval ARGS="--checkpoint outputs/2026-03-22/09-34-07/checkpoint_epoch0050.pt --num-clips 20 --batch-size 4 --device cpu"`
  - Result: `move_accuracy=0.0192`, `move_detection_f1=0.1124`, `avg_pgn_edit_distance=1.0000`, `avg_prefix_accuracy=0.0000`
  - The same run's training log already showed near-zero validation detection F1 across epochs, so the committed checkpoint itself is not good enough for an honest "model works" sign-off.

## Dev-tools regression fixes

- `/evaluate/overlay` no longer re-detects the overlay during the async FEN phase. The fast detect step now returns the exact refined board crop shown in the UI, and the classify/save calls reuse that same crop bytes.
- Added a local square-crop refinement pass for overlay eval crops so header/sidebar padding gets trimmed before FEN reads. This restores the previously good starting-position reads on frames like `Ez4friJ-mjo/25pct` and `uR2l9SNF1FU/50pct`.
- `/evaluate/fen` now samples committed board-crop fixtures only. That view is back to isolating piece-classifier accuracy; runtime full-frame real-data coverage stays on `/evaluate/overlay`.
- Regression protection added in `tests/dev_tools/test_overlay_test_service.py` for:
  - FEN fixture sampling ignoring ad-hoc local real crops
  - overlay detect phase returning a refined board crop, not the full overlay widget
  - async FEN classification reusing the provided crop instead of re-detecting from disk
  - saving confirmed extractions from the exact crop shown in the UI
- Validation after the fix:
  - `make typecheck`
  - `make lint`
  - `make test`

## Calibration follow-up fixes

- Shared OTB auto-calibration path now retries the OTB YOLO detector at higher input resolution on misses instead of giving up after the default 640px pass.
  - Root cause for `e4lGbQp4pU4` clip 70: overlay detection succeeded, but the OTB detector missed the small perspective board at 640 and recovered at 1280.
  - Added regression coverage in `tests/pipeline/test_otb_yolo_detector.py` for the retry path.
- Clip-level auto-calibration now returns failure diagnostics (`failure_reason`, detected overlay bbox, preview frame) so the UI can explain whether the miss was the overlay or the physical board.
- Calibration page UX fixes:
  - placeholder `0,0 100x100` camera boxes are treated as unset instead of being shown as if valid
  - the editor now snaps to the selected clip's start frame once the video session is ready
  - copy now tells the user to draw a tight blue box around the physical board only
  - auto-calibration preview colors now match the editor (`green=overlay`, `blue=OTB board`)
- Validation:
  - browser-validated `http://localhost:3000/videos/e4lGbQp4pU4/calibrate`
    - page now opens on clip start (`Frame 173 / 15446`, `5.8s`) instead of frame 0
    - `Auto-Calibrate This Clip` now succeeds for clip 70
    - resulting local clip calibration updated to overlay `[63, 6, 1045, 1074]`, camera `[1321, 878, 394, 179]`
  - backend-validated shared evaluation path on the same clip:
    - `api.services.evaluate.calibration_eval_service.inspect_calibration(70)` now returns `fresh_camera_bbox=[1320, 880, 396, 177]`
    - `camera_iou=0.9839`
- Validation caveat discovered while rerunning the full suite:
  - `make lint` ✅
  - `make typecheck` ✅
  - `make test` ❌ because `tests/pipeline/test_overlay_sequence_reader.py::test_full_board_read_microbenchmark` now measures ~`2.24s` vs the current `<1.6s` threshold on this machine. This is unrelated to the calibration changes but needs follow-up before claiming the whole branch is green again.


# Validation of progress tasks

To validate an item is concluded:
1. make tests, make lint, make types
2. you added new tests in @tests/ for the feature you implemented
3. if performance oriented you added a test with minimum performance threshold so it doesnt degrade
4. you updated @README.md and @CONTRIBUTING.md (if necessary)

To validate you finished all items in this list make sure:
1. tooling: the dev-tools ui is fully functioning
2. training: you can progress 5 of the videos (pick ones less than 200mb) we have and process them end to end, and output training data (pt files) with PGN,
timings, and board pictures
3. inference: make sure you validate against videos (or synthetic data) (i.e. val/) that our model works
