# Progress

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

# Validation of progress tasks

To validate an item is concluded:
1. make tests, make lint, make types
2. you added new tests in @tests/ for the feature you implemented
3. if performance oriented you added a test with minimum performance threshold so it doesnt degrade
4. you updated @README.md and @CONTRIBUTING.md (if necessary)
5. git commit and git push
6. pick up a new item to work on from progress.md, do not ask user for input just carry on

To validate you finished all items in this list make sure:
1. tooling: the dev-tools ui is fully functioning
2. training: you can progress 5 of the videos we have and process them end to end, and output training data (pt files) with PGN,
timings, and board pictures
3. inference: make sure you validate against videos (or synthetic data) (i.e. val/) that our model works

## Synthetic underfit investigation (2026-04-11)

- Wrote a technical report at:
  - `outputs/reports/2026-04-11-synthetic-underfit-report.md`
- Pre-training (unfreezing dino) seems like another thing to try further down the line
- Local diagnostic findings added beyond the training logs:
  - synthetic clips are labeled as **change points**:
    - positive `detect_targets` mark the first stable post-move frame
    - later frames with the exact same FEN are labeled negative
  - measured over sampled synthetic clips:
    - average positive-frame fraction was ~`0.38`
    - average legal-move count on move frames was ~`25.7` to `32.5` depending on sample
    - so the detector is **not** failing because positives are vanishingly rare
  - feature-space probe with `facebook/dinov2-small` on an actually rendered augmented synthetic clip:
    - mean-pooled frame embeddings had only a tiny same-vs-change separation
      - same-FEN cosine ≈ `0.9775`
      - changed-FEN cosine ≈ `0.9716`
      - gap ≈ `0.0059`
    - flattened patch-token features preserved a materially larger gap
      - same-FEN cosine ≈ `0.8689`
      - changed-FEN cosine ≈ `0.8330`
      - gap ≈ `0.0359`
    - implication:
      - the current `VisionEncoder.forward_pooled()` path is compressing away much of the sparse square-local move signal
  - detector-collapse probe on `outputs/2026-04-11/synth_control_small/checkpoint_epoch0004.pt`:
    - mean predicted detect probability on positives ≈ `0.3601`
    - mean predicted detect probability on negatives ≈ `0.3577`
    - this checkpoint mostly learned the class prior instead of true change-point detection
- Code-level synthetic-path findings from the report:
  - `scripts/train.py` / `scripts/evaluate.py` do **not** wire through several knobs present in `configs/data/synthetic.yaml`
    - `pgn_source`
    - `min_elo`
    - `augmentations.*`
    - `frames_per_move`
    - `occlusion_prob`
  - `src/argus/datagen/synth.py` currently samples uniformly random legal games via `sample_random_game()` instead of using the configured PGN source, making the synthetic task harsher and less realistic than the config suggests
  - `src/argus/data/transforms.py` defines ImageNet-style normalization for DINO inputs, but the normal training/eval path does not currently attach those transforms to the datasets
- Oracle review requested via the `oracle` skill after the report was written.
- Oracle feedback that appears valid:
  - missing DINO input normalization is likely a bigger issue than the report initially ranked it
  - synthetic-control evidence from only `64` optimizer steps should not be treated as a decisive convergence limit
- Current post-oracle assessment:
  - normalization should be treated as a top-tier issue, not a footnote
  - however, the local feature probes still support the original conclusion that global mean pooling is also a real representation bottleneck even **after** normalization, because the normalized pooled same-vs-change gap remained very small compared with patch-level features
  - best next work should focus on a small number of targeted debugging experiments rather than more broad real-data collection:
    1. wire normalization into the train/eval path and rerun tiny synthetic overfit + longer synthetic control
    2. expose synthetic-generation knobs through the CLI/train entrypoint so augmentation and PGN realism can be simplified deliberately
    3. test whether a patch-aware / spatially preserved board representation learns change points materially better than the current mean-pooled frame embedding path
- Follow-up implementation and experiments completed after that assessment:
  - device selection is now centralized in `src/argus/device.py` and consistently prefers:
    1. `cuda`
    2. `mps`
    3. `cpu`
  - the preference order is now used by:
    - `scripts/train.py`
    - `scripts/evaluate.py`
    - `scripts/infer.py`
    - `scripts/eval_chess_positions.py`
    - `scripts/train_piece_classifier.py`
    - `src/argus/training/trainer.py`
    - `src/argus/eval/evaluator.py`
    - `src/argus/inference/pipeline.py`
  - MPS smoke validation succeeded:
    - `.venv/bin/python3 scripts/train.py model=argus_small data=synthetic data.num_train_clips=4 data.num_val_clips=1 data.clip_length=8 data.image_size=64 data.augment=false training.epochs=1 training.batch_size=2 training.gradient_accumulation=1 training.wandb.enabled=false output_dir=outputs/2026-04-11/mps_device_smoke`
    - runtime picked `mps` automatically and trained successfully
  - synthetic train/eval path now always applies DINO input normalization via `ValidationTransform()`
  - synthetic config now matches actual train/eval wiring much more closely:
    - new/used knobs include:
      - `frames_per_move`
      - `augment`
      - `occlusion_prob`
      - `game_source`
      - `pgn_path`
      - `min_elo`
      - `min_moves`
      - `max_moves`
    - `src/argus/datagen/synth.py` now supports `game_source="pgn_file"`
    - regression coverage added in `tests/test_synth.py`
  - added patch-token pooling abstraction in `src/argus/model/patch_pooling.py`
    - `mean`
    - `square_attention`
  - `ArgusModel` now uses patch tokens plus configurable pooling instead of hardcoded `forward_pooled()` mean pooling
  - pooling experiment results documented in:
    - `outputs/reports/2026-04-11-pooling-followup.md`
  - important evaluation correction landed:
    - `compute_move_metrics()` now respects `detect_threshold`
    - earlier all-zero detection-F1 conclusions at threshold `0.5` were partly an evaluation artifact
    - regression coverage added in `tests/test_eval_metrics.py`
  - pooled synthetic-control comparison after normalization + threshold-fix:
    - smaller probe (`64 train / 16 val`, `160` optimizer steps):
      - mean pooling final val move acc: `0.0435`
      - square-attention final val move acc: `0.0281`
      - square-attention fit train better but was not clearly better on held-out data at that tiny scale
    - less tiny control (`128 train / 32 val`, `240` optimizer steps):
      - mean pooling final val move acc: `0.0497`
      - square-attention final val move acc: `0.0577`
      - fresh synthetic eval at threshold `0.3`:
        - mean pooling:
          - `move_acc=0.0408`
          - `detect_f1=0.5560`
          - `avg_pgn_edit_distance=0.9823`
        - square-attention:
          - `move_acc=0.0421`
          - `detect_f1=0.5556`
          - `avg_pgn_edit_distance=0.9772`
      - takeaway:
        - square-aware pooling is a **modest** improvement, not a fix
        - normalization + honest detect-threshold handling mattered more than the original mean-vs-square comparison suggested
  - used `oracle` again after the pooling follow-up:
    - oracle conclusion:
      - the model can detect that *something changed*, but still cannot read the board well enough
      - next best test is dense board-state supervision, not more pooling variants or a backbone swap
  - implemented dense board-state supervision path:
    - new file `src/argus/chess/board_state.py`
      - `fen_to_square_targets(...)`
    - synthetic clips now persist `board_flipped`
    - datasets now derive optional `square_targets` from stored `fens`
    - `src/argus/model/square_head.py`
      - per-square board-state classifier
    - `ArgusModel` now optionally emits `square_logits`
    - `ArgusLoss` now supports `square` loss
    - trainer/validation now log square-state accuracy when present
    - added regression coverage:
      - `tests/test_board_state.py`
      - expanded `tests/test_argus_dataset.py`
      - expanded `tests/test_argus_model.py`
  - dense square-state probe documented in:
    - `outputs/reports/2026-04-11-square-state-probe.md`
  - first dense board-state probe (frozen DINO):
    - command:
      - `.venv/bin/python3 scripts/train.py model=argus_small model.pooling.type=mean model.square_head.enabled=true data=synthetic data.num_train_clips=64 data.num_val_clips=16 data.clip_length=32 data.augment=false training.epochs=10 training.batch_size=8 training.gradient_accumulation=1 training.optimizer.lr=1e-4 training.loss_weights.move=0 training.loss_weights.detect=0 training.loss_weights.square=1.0 training.wandb.enabled=false output_dir=outputs/2026-04-11/synth_square_state_probe`
    - result:
      - square accuracy quickly rose above chance but plateaued low:
        - train `square_acc≈0.5325`
        - val `square_accuracy≈0.5419`
      - interpretation:
        - dense board-state supervision is easier than direct move prediction
        - but frozen DINOv2-small plus the current simple square readout is still **not** good enough to read boards accurately
  - oracle review of that probe concluded the next clean test is limited unfreezing of the vision encoder before attempting a new backbone
  - implemented training-time support for `vision_encoder.unfreeze_last_n`
  - follow-up square-state probe with `unfreeze_last_n=3`:
    - command:
      - `.venv/bin/python3 scripts/train.py model=argus_small model.vision_encoder.unfreeze_last_n=3 model.pooling.type=mean model.square_head.enabled=true data=synthetic data.num_train_clips=64 data.num_val_clips=16 data.clip_length=32 data.augment=false training.epochs=10 training.batch_size=8 training.gradient_accumulation=1 training.optimizer.lr=1e-4 training.loss_weights.move=0 training.loss_weights.detect=0 training.loss_weights.square=1.0 training.wandb.enabled=false output_dir=outputs/2026-04-11/synth_square_state_unfreeze3`
    - result:
      - no clear improvement over frozen-DINO probe in this small run
      - val `square_accuracy` stayed in the same rough band (`~0.53-0.54`)
      - implication:
        - simply unfreezing the last 3 layers with the current setup is **not** an immediate breakthrough
        - the next backbone/readout decision should be made deliberately rather than assuming shallow fine-tuning solves it
- Current state after these iterations:
  - the original underfit story has been refined:
    - missing normalization was real
    - detect-threshold evaluation was misleading
    - square-aware pooling helps a bit but is not the main fix
    - dense board-state supervision is the right conceptual direction
    - however, the current frozen DINOv2-small representation still appears too weak for reliable board reading, and a small last-layer unfreeze did not obviously fix that
  - best next likely branch of work, if continuing from here:
    1. improve the dense board-state path further (better square readout / spatial alignment and a stronger square-state benchmark)
    2. only then decide whether to unfreeze more aggressively or try a different backbone such as a DINOv3-family model
