# Contributing to Argus

## Prerequisites

| Tool | Version | Purpose | Install (macOS) |
|------|---------|---------|-----------------|
| Python | 3.10+ | ML code and pipeline | `brew install python@3.10` |
| Node.js | 18+ | Dev tools web UI | `brew install node` |
| Docker | Latest | PostgreSQL + dev tools | `brew install --cask docker` |
| Git LFS | Latest | Committed model weights under `weights/` | `brew install git-lfs` |
| Cairo | Latest | SVG rendering for synthetic data | `brew install cairo` |
| Blender | 4.0+ | 3D synthetic data rendering | `brew install --cask blender` |
| ffmpeg | Latest | Frame extraction from videos | `brew install ffmpeg` |

<details>
<summary>Linux install instructions</summary>

**Cairo:**
```bash
# Ubuntu / Debian
sudo apt-get install libcairo2-dev
# Fedora
sudo dnf install cairo-devel
```

**Blender:**
```bash
# Ubuntu / Debian
sudo snap install blender --classic
# Or download from https://www.blender.org/download/
```

**ffmpeg:**
```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg
```
</details>

Verify Blender: `blender --version` should show 4.0 or later. You can also set the `BLENDER_PATH` environment variable to point to a custom Blender install.

---

## Which Domain?

| I want to... | Domain | Folder | Setup needed |
|--------------|--------|--------|-------------|
| Add YouTube channels or improve video screening | Pipeline | `pipeline/` | Python + PostgreSQL + API keys |
| Improve AI screening classifier accuracy | Pipeline (screen) | `pipeline/screen/ai_*.py` | Python + PyTorch + labelled videos in DB |
| Improve the default YOLO overlay detector or calibration | Pipeline (overlay) | `pipeline/overlay/` | Python + Cairo |
| Tune auto-calibration (theme/orientation detection) | Pipeline (overlay) | `pipeline/overlay/auto_calibration.py` | Python + Cairo |
| Generate better synthetic training data | Data Generation | `src/argus/datagen/` | Python + Cairo (+ Blender 4.0+ for 3D) |
| Train the model or tune hyperparameters | Training | `src/argus/model/`, `src/argus/training/` | Python + PyTorch + GPU |
| Improve inference accuracy or add streaming | Inference | `src/argus/inference/` | Python + PyTorch + trained checkpoint |
| Build or improve developer inspection tools | Dev Tools | `dev-tools/` | Docker (or Python + Node.js) |

---

## Development Setup

### Common (all domains)

```bash
git clone <repo-url> && cd argus
git lfs install
git lfs pull --include="weights/screening/*,weights/overlay/*,weights/overlay_yolo/*"
cp .env.example .env    # fill in HF_TOKEN, API keys, etc.

# Option 1: direnv (recommended) — auto-activates venv + loads .env on cd
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc  # or ~/.bashrc
direnv allow  # first time only

# Option 2: manual
python3 -m venv .venv && source .venv/bin/activate
make dev
```

`make dev` installs the host-side YOLO runtime dependency (`ultralytics`). `make up` installs the same dependency in the API container from `pipeline/requirements.txt`.

Start all services (PostgreSQL, dev-tools API, dev-tools UI via Docker; Blender natively) in the background:

```bash
make up       # start everything
make down     # stop everything
```

`make up`, `make test`, and the runtime pipeline targets now fail fast if committed model weights are missing or still Git LFS pointers.

### Pipeline

```bash
make pipeline-install   # installs pipeline/ dependencies (psycopg, yt-dlp, etc.)
```

PostgreSQL is already running via `make up`.

### Data Generation

No extra setup beyond `make dev`. Cairo must be installed system-wide (see Prerequisites).

For 3D data, install Blender 4.0+ (see Prerequisites). The STL piece models are included in `blender/models/staunton/`. Set `BLENDER_PATH` if Blender is not on your PATH. `make up` starts the Blender render server automatically (skips with a warning if Blender is not installed).

### Training

```bash
# CPU/MPS (default — uses GRU fallback for temporal module)
make dev

# GPU with Mamba-2 (requires CUDA)
pip install -e ".[cuda,dev]"
```

### Dev Tools

Already running via `make up`. For foreground mode with streaming logs: `make dev-tools`.

For manual dev-tools development (docker is preferred):

```bash
# Terminal 1:
cd dev-tools/api && python -m uvicorn main:app --reload --port 8000
# Terminal 2:
cd dev-tools && npm install && npm run dev
```

---

## Running Tests

### Per-Domain Test Commands

| Domain | Command | What it tests |
|--------|---------|--------------|
| **All** | `make test` | Full suite |
| **Chess core** | `pytest tests/test_move_vocabulary.py tests/test_chess_state_machine.py tests/test_constraint_mask.py -v` | Move vocabulary, state machine, constraint masking |
| **Pipeline — overlay** | `pytest tests/pipeline/test_overlay_move_detector.py -v` | Overlay move detection (including hard cut detection) |
| **Pipeline — screening** | `pytest tests/pipeline/ -v` | Screening orchestration |

### Quick Smoke Tests (no DB required)

```bash
# Run all smoke tests (hard cut detection + AI classifier)
python -m pipeline smoke-test

# Or in Docker:
make docker-smoke-test
```

### Linting & Type Checking

```bash
make lint       # ruff check on src/, tests/, scripts/
make typecheck  # mypy on src/argus/
make format     # ruff format + auto-fix (run before committing)
```

---

## Code Style

- **Formatter/linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Line length**: 100
- **Target**: Python 3.10
- **Lint rules**: E, F, I (isort), N (naming), W, UP (pyupgrade)
- **Type checking**: mypy strict mode

Run `make format` before committing.

---

## How to Add a New Pipeline Stage

1. Create a module under `pipeline/<stage_name>/`
2. Add the main function (e.g., `run_all()`)
3. Register the CLI command in `pipeline/cli.py`:
   - Add a `cmd_<name>(args)` function
   - Add the subparser in `main()`
   - Add the dispatch entry in the `commands` dict
4. Add a Makefile target if it's a standard pipeline step
5. Add tests under `tests/pipeline/`

---

## Validating the Pipeline Features

Each pipeline feature can be validated end-to-end without processing the full dataset. Use the dev-tools web UI (localhost:3000) alongside the CLI for visual inspection.

### Validating Hard Cut Detection

Hard cut detection splits clip segments when a video abruptly changes between games without resetting to the starting position.

```bash
# 1. Pick a video known to contain mid-stream game switches
#    (check videos from multi-board tournament channels like @STLChessClub)

# 2. Run clip generation and check segment count
python -m pipeline generate-clips --channel @STLChessClub --limit 1 -v

# 3. Inspect a generated clip — verify move_confidence is present
python -m pipeline inspect-clip --file data/argus/training_clips/overlay_VIDEO_ID_0.pt

# 4. Use the dev-tools Video Annotator (localhost:3000/videos/VIDEO_ID):
#    - Open the video and run "Detect Moves"
#    - Check that each move in the move list shows a confidence score
#    - Verify that game segments are split at hard cuts (multiple segments shown)
```

### Validating the Default YOLO Overlay Detector

The runtime overlay detector is YOLO-based. Bounding boxes stored in
`data/videos/ground_truth.json` and `tests/fixtures/frames/ground_truth.json`
exist only to train and evaluate that detector.

```bash
# 1. Export the current training labels
python -m pipeline overlay-yolo-export

# 2. Train a detector
python -m pipeline overlay-yolo-train --epochs 40 --batch 8

# 3. Visualize detector output on the committed fixture set
#    Run this after every detector change.
.venv/bin/python3 scripts/visualize_overlay_tests.py --out-dir outputs/overlay_test_viz

# 4. Full validation
make typecheck
make lint
make test
```

If you touch detector code, do not ship a change that only looks good in unit
metrics. The contact sheet from `scripts/visualize_overlay_tests.py` is the
source of truth for whether the detector actually improved.

### Validating Auto-Calibration

Auto-calibration proposes overlay/camera crop regions, theme, and board orientation from YouTube thumbnails.

```bash
# 1. Test against a channel with existing manual calibration
python -m pipeline auto-calibrate --channel @STLChessClub

# 2. Compare the proposed values against the manual calibration:
python -m pipeline inspect-calibration --channel @STLChessClub

# 3. Overlay/camera bboxes should be within ~10% of manual values
#    Theme and orientation should match exactly

# 4. Use the dev-tools API directly:
#    POST http://localhost:8000/api/calibration/@STLChessClub/propose
#    Compare returned proposal against existing calibration

# 5. For a new channel, apply and verify:
python -m pipeline auto-calibrate --channel @NewChannel --apply
```

### Validating AI Screening

The AI screening classifier automates the 3-way video review (overlay / otb_only / reject). Vertical videos are auto-rejected. Pre-trained weights are committed in `weights/screening/`.

#### Model versioning

Versions follow the format `v{code}r{revision}` (e.g. `v2r3`):
- **Code version** (`MODEL_CODE_VERSION` in `ai_classifier.py`): bump when architecture or feature extraction changes
- **Revision**: auto-incremented on each training run

Weights are saved to `weights/screening/{version}.pt` and should be committed to the repo so the model is always available without retraining.

#### Training workflow

```bash
# 1. Run the DB migration first (adds screening columns)
#    This runs automatically on API startup, or manually:
#    psql $DATABASE_URL -f pipeline/db/migrations/002_add_ai_screening.sql

# 2. Pre-compute DINOv2 features for all labelled videos
#    This is the slow step (~4s/video, network-bound). Resumable — skips cached.
#    Features are cached in data/screening/dataset/torch/ (not committed).
python -m pipeline ai-extract --device mps  # use 'cuda' on Linux, 'cpu' as fallback

# 3. Train the classifier head (fast, ~30s — operates on cached features)
#    Saves weights to both data/screening/checkpoints/ (ephemeral) and weights/screening/ (committed)
python -m pipeline ai-train --epochs 50

# 4. Evaluate and calibrate the confidence threshold
python -m pipeline ai-eval --target-precision 0.95

# 5. Verify in the dev-tools UI
#    Evaluate > Screening > "Sample & Inspect" — runs model on 20 random
#    labeled videos and shows accuracy, model vs human label per video,
#    title score, and vertical detection

# 6. Run AI screening on unscreened videos
python -m pipeline ai-screen --limit 10 --threshold 0.90

# 7. Commit the new weights
git add weights/screening/
git commit -m "screening: train v2r3 (95% accuracy on N samples)"
```

#### Running in Docker

```bash
# Extract, train, evaluate (transformers is pre-installed in the Docker image)
make docker-ai-extract ARGS="--device cpu"
make docker-ai-train ARGS="--epochs 50 --device cpu"
make docker-ai-eval

# Check extraction progress
make docker-ai-extract-status
```

### Integration with Dev Tools

| Feature | Where to verify in dev-tools |
|---------|------------------------------|
| **Hard cuts** | `/videos/VIDEO_ID` → run "Detect Moves" → check segment count and per-move confidence in the move list |
| **YOLO overlay detector** | `Annotate > BBox` for training labels, plus `scripts/visualize_overlay_tests.py` for the actual detector regression view |
| **Auto-calibration** | `POST /api/calibration/{channel}/propose` → compare returned bboxes with manual calibration. Or use the Annotate > Calibrate page to view/edit proposed values |
| **AI screening** | Evaluate > Screening > "Sample & Inspect" — shows model prediction vs human label per video with accuracy summary. Also available per-video at `/videos/VIDEO_ID` > Info > "Run AI Screen" |

---

## How to Add a New Dev Tool

The dev tools follow a **router -> service -> pipeline module** pattern organized by section. Every web tool should have a CLI equivalent.

1. **Service** (`dev-tools/api/services/{videos,annotate,data,evaluate}/<name>_service.py`): Business logic wrapping `pipeline.*` modules
2. **Router** (`dev-tools/api/routers/{videos,annotate,data,evaluate}/<name>.py`): FastAPI endpoints calling the service
3. **Register** in `dev-tools/api/main.py`: `app.include_router()`
4. **Page** (`dev-tools/app/{videos,annotate,data,evaluate}/<name>/page.tsx`): React page consuming the API
5. **Navigation**: Add tab to the section layout or sidebar item in `dev-tools/components/IconSidebar.tsx`
6. **CLI parity**: Add or verify a matching CLI command in `pipeline/cli.py`

Reusable UI components live in `dev-tools/components/` — check `ChessBoard`, `FileUpload`, and `video-shared` before building new ones. Section-specific components are in `dev-tools/components/{videos,annotate,data,evaluate}/`.
