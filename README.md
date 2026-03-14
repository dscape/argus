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
python -m venv .venv
source .venv/bin/activate
make dev        # install with dev dependencies
make test       # run tests
```

Requires Python 3.10+ and a CUDA-capable GPU for training.

## Usage

### Training

```bash
# Phase 1: Single-board move recognition on 2D synthetic data
make train ARGS="training=phase1_detection"

# Phase 2: Multi-board detection + recognition on 3D synthetic data
make train ARGS="training=phase2_recognition"

# Phase 3: End-to-end fine-tuning with curriculum
make train ARGS="training=phase3_endtoend"
```

Training is configured via [Hydra](https://hydra.cc/) YAML configs in `configs/`. Override any parameter from the command line:

```bash
make train ARGS="training=phase1_detection training.batch_size=16 training.lr=5e-4"
```

### Synthetic Data Generation

```bash
# 2D sprite-based data (fast, for Phase 1)
make datagen ARGS="--mode synth2d --num-clips 1000 --output-dir data/synth2d"

# 3D Blender-rendered data (requires Blender 4.0+, for Phase 2+)
make datagen ARGS="--mode blender --config scene_tournament --num-clips 200"
```

### Inference

```bash
# Single-board video → PGN
make infer ARGS="--video game.mp4 --output game.pgn"

# Multi-board tournament video
make infer ARGS="--video tournament.mp4 --output-dir pgns/ --checkpoint model.pt"
```

### Evaluation

```bash
make eval ARGS="--checkpoint model.pt --data-dir data/test"
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
