# Progress

## Current architecture findings (2026-04-08)

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

## Dev tools debugging notes (2026-04-08)

- Reproduced the extract-page failure on `/videos/9h4IE1G99OE/extract`: the `Run Detection` click was firing, but the synchronous `POST /api/video/{session}/detect-moves` request sat behind the Next.js dev proxy until it failed with a 500 after roughly 5 minutes.
- Root cause: move detection is a long-running CPU job in the dev-tools API, while the UI was waiting on a single proxied request. The button was not dead; the request was timing out at the proxy boundary before results could come back.
- Implemented a background move-detection job flow for the extract page:
  - added API endpoints to start a detection job and poll job status
  - updated the extract UI to kick off detection asynchronously, show in-progress state, and render results once polling reports `done`
  - kept the existing synchronous detect endpoint in place for direct callers
- Validation:
  - `tests/dev_tools/test_video_annotation_service.py -q` passes with coverage for the new job lifecycle
  - `make typecheck` passes
  - `make lint` passes
  - `make test` passes
  - live proxy verification: starting detection now returns immediately with a job id and `status: "running"` instead of hanging

## Overlay bbox annotation notes (2026-04-08)

- Updated `/annotate/bbox` so loading an unannotated frame now triggers a best-effort overlay suggestion automatically.
- Implementation details:
  - added a dev-tools API endpoint that runs the committed YOLO overlay detector on the selected frame
  - auto-detected boxes are lightly normalized through the existing square/grid-alignment refinement helper before they are shown in the UI
  - the page only auto-fills unannotated frames; saved annotations still load exactly as stored
  - manual drawing remains the fallback and takes precedence over late detector responses
  - `Save` and `No Overlay` now immediately advance to the next remaining unannotated frame, and the standalone `Next` button was removed
- Validation:
  - `python3 -m pytest tests/dev_tools/test_overlay_bbox_service.py -q` via `.venv/bin/python3` passes
  - `make typecheck` passes
  - `make lint` passes
  - `make test` passes

## Pending performance work (2026-04-08)

- Cheap move-change gate for extract-page detection is still missing.
  - Current state: move detection still runs a full `read_overlay_crop(...)` on every sampled frame.
  - Example cost: clip `69` for video `9h4IE1G99OE` produces `376` sampled frames at `2.0` FPS, and each sample pays for full-board reading work.
  - Planned optimization:
    - lock board geometry once from clip calibration instead of rediscovering it every sample
    - compute cheap per-square visual deltas between samples
    - only trigger expensive FEN reading on stable candidate transitions, likely when `2-4` squares change or on periodic resync frames
  - Ask for tests when implementing:
    - add unit tests for the per-square change gate and debounce / hysteresis logic
    - add integration coverage showing that non-move frames are skipped while legal moves are still detected
    - add regression coverage for castling, captures, noisy highlight animations, and false-positive avoidance

- 64-square DINOv2 board reading path is much slower than it should be and needs dedicated optimization.
  - Current concern: the overlay reader classifies all `64` squares through the DINOv2-based path and this is a major runtime bottleneck in extraction.
  - Planned optimization areas:
    - profile preprocessing vs encoder time vs per-square batching overhead
    - confirm the `64`-square batch is actually using the fastest available path on CPU / MPS
    - avoid redundant preprocessing and repeated grid work where the board geometry is stable
    - consider lighter-weight square embeddings or a cheaper candidate filter before invoking the full classifier
  - Ask for tests and benchmarks when implementing:
    - add a repeatable microbenchmark for full 64-square board reads
    - add before/after timing assertions or at least recorded benchmark output in `progress.md`
    - keep accuracy regression tests for FEN extraction so speed work does not silently degrade board reads
    - add more robust tests on failing cases and fix them. Running /evaluate/screening and /evaluate/fen often produce mistakes. Find at least 10 such mistakes and make sure they pass as tests in fixtures
