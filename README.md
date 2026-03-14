# Argus

Multi-game chess board state tracking from unconstrained video.

Argus reconstructs PGN game records from tournament video by framing move recognition as a VLA-style sequential decision problem. A single model observes video frames and emits `(board_id, move)` events, with chess legality enforced architecturally through constrained decoding — the model literally cannot output an illegal move.

## Architecture

```
Video Frame → DINOv2 ViT-B/14 → Board Detector (DETR-style) → Per-board crops
                                                                    ↓
Per-board features → Mamba-2 SSM (temporal memory) → Constrained Move Head → (board_id, move_uci)
                                                           ↑
                                                 Legal move mask (python-chess)
```

**Vision Encoder** — DINOv2 ViT-B/14 (frozen, then fine-tuned) provides dense spatial features for both board detection and piece recognition.

**Board Detector** — DETR-style transformer decoder with learned board queries. Outputs bounding boxes and identity embeddings, tracked across frames via Hungarian matching on cosine similarity.

**Temporal Module** — Mamba-2 SSM processes per-board feature sequences with linear-time complexity, handling 4+ hour tournaments. Hidden state acts as implicit game memory.

**Constrained Move Head** — Projects to 1970 logits (1968 UCI moves + NO_MOVE + UNKNOWN). A legal move mask from python-chess zeros out illegal moves before softmax, guaranteeing only legal moves are predicted.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make dev        # install with dev dependencies
make test       # run tests (34 tests)
```

Requires Python 3.10+. For training with Mamba-2 (instead of the GRU fallback), install CUDA dependencies on a GPU machine:

```bash
pip install -e ".[cuda,dev]"
```

## Usage

### Synthetic Data Generation

Generate training data to disk first — this is much faster than generating on-the-fly during training:

```bash
# 2D sprite-based data (Phase 1)
make datagen ARGS="--num-clips 5000 --output-dir data/train"
make datagen ARGS="--num-clips 500 --output-dir data/val"

# Smaller/faster for local development
make datagen ARGS="--num-clips 100 --output-dir data/dev --image-size 64"

# 3D Blender-rendered data (Phase 2+, requires Blender 4.0+)
make datagen ARGS="--type 3d --num-clips 200 --output-dir data/3d"
```

### Training

```bash
# Train from pre-generated data on disk (recommended)
make train ARGS="data.data_dir=data training.wandb.enabled=false"

# Or generate on-the-fly (slower, no prep step needed)
make train ARGS="data.num_train_clips=100 data.num_val_clips=20 data.image_size=64 training.wandb.enabled=false"

# Phase configs
make train ARGS="training=phase1_detection data.data_dir=data"
make train ARGS="training=phase2_recognition data.data_dir=data"
make train ARGS="training=phase3_endtoend data.data_dir=data"
```

Training is configured via [Hydra](https://hydra.cc/) YAML configs in `configs/`. Override any parameter from the command line:

```bash
make train ARGS="training=phase1_detection training.batch_size=16 training.optimizer.lr=5e-4"
```

### Inference

```bash
make infer ARGS="--video tournament.mp4 --checkpoint outputs/checkpoint_epoch0050.pt --output-dir pgns/"
```

### Evaluation

```bash
make eval ARGS="--checkpoint outputs/checkpoint_epoch0050.pt --num-clips 200"
```

## Project Structure

```
argus/
├── configs/                        # Hydra YAML configs
│   ├── config.yaml                 # Root config
│   ├── model/                      # argus_base.yaml, argus_small.yaml
│   ├── data/                       # synthetic.yaml, real.yaml
│   ├── training/                   # phase1, phase2, phase3
│   ├── eval/                       # default.yaml
│   └── datagen/                    # scene configs
├── src/argus/
│   ├── types.py                    # Core dataclasses
│   ├── chess/                      # Chess logic layer
│   │   ├── move_vocabulary.py      # 1968 UCI moves + special tokens
│   │   ├── state_machine.py        # python-chess wrapper, legal mask generation
│   │   ├── constraint_mask.py      # Legal move masking for model output
│   │   └── pgn_writer.py           # Move events → PGN
│   ├── model/                      # Neural network components
│   │   ├── vision_encoder.py       # DINOv2 ViT-B/14
│   │   ├── board_detector.py       # DETR-style detection
│   │   ├── board_id_head.py        # Board identity tracking
│   │   ├── temporal.py             # Mamba-2 SSM
│   │   ├── move_head.py            # Constrained move prediction
│   │   ├── losses.py               # Focal + CE + GIoU + contrastive
│   │   └── argus_model.py          # Full model assembly
│   ├── data/                       # Data loading
│   │   ├── dataset.py              # PyTorch Dataset
│   │   ├── transforms.py           # Augmentations
│   │   ├── collate.py              # Variable-length batching
│   │   └── pgn_sampler.py          # Game sampling from PGN files
│   ├── datagen/                    # Synthetic data generation
│   │   ├── synth2d.py              # 2D sprite compositing
│   │   ├── scene_builder.py        # Blender scene composition
│   │   ├── camera.py               # Camera placement/motion
│   │   ├── lighting.py             # Lighting variation
│   │   ├── humans.py               # Occlusion simulation
│   │   ├── game_driver.py          # PGN → 3D piece positions
│   │   └── renderer.py             # Render loop + annotations
│   ├── training/                   # Training loop
│   │   ├── trainer.py              # Training with wandb, bf16, grad accum
│   │   └── scheduler.py            # Curriculum learning
│   ├── eval/                       # Evaluation
│   │   ├── metrics.py              # MA, MDF1, PED, PA, ISR, ORR
│   │   ├── evaluator.py            # End-to-end eval pipeline
│   │   └── visualizer.py           # Prediction overlay on video
│   └── inference/                  # Runtime inference
│       ├── pipeline.py             # Video → PGN
│       ├── tracker.py              # Multi-game tracker with beam search
│       └── postprocess.py          # Confidence gating, game completion
├── scripts/                        # CLI entry points
├── tests/                          # pytest suite
└── pyproject.toml
```

## Key Design Decisions

**Constrained decoding over post-hoc filtering.** The legal move mask is applied before softmax, not after. The model's probability distribution is defined only over legal moves, so training signal is never wasted on impossible outputs.

**Move vocabulary as fixed enumeration.** All 1968 reachable UCI moves (queen/rook/bishop lines + knight L-shapes + pawn promotions) are assigned deterministic indices. This mapping never changes — model weights, loss functions, and metrics all depend on it.

**Mamba-2 over transformers for temporal modeling.** Linear-time complexity in sequence length handles full tournaments (14K+ frames) without quadratic attention costs. The SSM hidden state acts as compressed game memory.

**Synthetic data first.** 2D sprite compositing enables rapid iteration on model architecture before investing in expensive Blender renders. The curriculum progressively increases difficulty (resolution, occlusion, board count).

## Metrics

| Metric | Description |
|--------|-------------|
| Move Accuracy (MA) | Correct moves / total moves |
| Move Detection F1 (MDF1) | Precision/recall on "did a move happen?" |
| PGN Edit Distance (PED) | Levenshtein distance between predicted and GT move lists |
| Prefix Accuracy (PA) | Longest correct PGN prefix / game length |
| Board Detection mAP | Standard mAP@0.5 for board localization |
| Identity Switch Rate (ISR) | ID switches per 1000 frames |
| Occlusion Recovery Rate (ORR) | Correct re-ID after N frames of occlusion |

## License

MIT
