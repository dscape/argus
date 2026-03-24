# Contributing to Argus

## Prerequisites

| Tool | Version | Purpose | Install (macOS) |
|------|---------|---------|-----------------|
| Python | 3.10+ | ML code and pipeline | `brew install python@3.10` |
| Node.js | 18+ | Dev tools web UI | `brew install node` |
| Docker | Latest | PostgreSQL + dev tools | `brew install --cask docker` |
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
| Improve overlay board reading or calibration | Pipeline (overlay) | `pipeline/overlay/` | Python + Cairo |
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
cp .env.example .env    # fill in HF_TOKEN, API keys, etc.

# Option 1: direnv (recommended) — auto-activates venv + loads .env on cd
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc  # or ~/.bashrc
direnv allow  # first time only

# Option 2: manual
python3 -m venv .venv && source .venv/bin/activate
make dev
```

Start all services (PostgreSQL, dev-tools API, dev-tools UI, Blender) in the background:

```bash
make up       # start everything
make down     # stop everything
```

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

For manual dev-tools development:

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
| **Pipeline — overlay** | `pytest tests/pipeline/test_overlay_reader.py tests/pipeline/test_overlay_move_detector.py -v` | Overlay FEN reading, move detection (including hard cut detection) |
| **Pipeline — screening** | `pytest tests/pipeline/test_screen_pipeline.py tests/pipeline/test_title_filter.py -v` | Title filter, screening orchestration |

### Quick Smoke Tests (no DB required)

```bash
# Verify hard cut detection logic
python -c "
from pipeline.overlay.overlay_move_detector import count_fen_differences
import chess
# Starting position vs empty board — should be 32 (all pieces changed)
print('Full reset:', count_fen_differences(chess.STARTING_BOARD_FEN, '8/8/8/8/8/8/8/8'))
# e2e4 — should be 2 (pawn moved from e2 to e4)
print('e2e4:', count_fen_differences(chess.STARTING_BOARD_FEN, 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR'))
"

# Verify AI classifier model can be instantiated
python -c "
from pipeline.screen.ai_classifier import ScreeningClassifier
import torch
model = ScreeningClassifier()
# Fake input: 1 video, 4 frames, 768-dim embeddings + scores
emb = torch.randn(1, 4, 768)
scan = torch.randn(1, 4)
otb = torch.randn(1, 4)
logits = model(emb, scan, otb)
print('Logits shape:', logits.shape)  # Should be (1, 3)
print('Classes: overlay, otb_only, reject')
"
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
python -c "
import torch
clip = torch.load('data/training_clips/overlay_VIDEO_ID_0.pt', weights_only=True)
print('Keys:', list(clip.keys()))
print('move_confidence shape:', clip['move_confidence'].shape)
print('move_confidence range:', clip['move_confidence'].min().item(), '-', clip['move_confidence'].max().item())
print('Low-confidence moves:', (clip['move_confidence'] < 1.0).sum().item())
"

# 4. Use the dev-tools Video Annotator (localhost:3000/videos/VIDEO_ID):
#    - Open the video and run "Detect Moves"
#    - Check that each move in the move list shows a confidence score
#    - Verify that game segments are split at hard cuts (multiple segments shown)
```

### Validating Auto-Calibration

Auto-calibration proposes overlay/camera crop regions, theme, and board orientation from YouTube thumbnails.

```bash
# 1. Test against a channel with existing manual calibration
python -m pipeline auto-calibrate --channel @STLChessClub

# 2. Compare the proposed values against the manual calibration:
python -c "
from pipeline.overlay.calibration import get_calibration
cal = get_calibration('@STLChessClub')
print('Manual overlay:', cal.overlay)
print('Manual camera:', cal.camera)
print('Manual theme:', cal.board_theme)
print('Manual flipped:', cal.board_flipped)
"

# 3. Overlay/camera bboxes should be within ~10% of manual values
#    Theme and orientation should match exactly

# 4. Use the dev-tools API directly:
#    POST http://localhost:8000/api/calibration/@STLChessClub/propose
#    Compare returned proposal against existing calibration

# 5. For a new channel, apply and verify:
python -m pipeline auto-calibrate --channel @NewChannel --apply
```

### Validating AI Screening

The AI screening classifier automates the 3-way video review (overlay / otb_only / reject). Vertical videos are auto-rejected. Pre-trained weights are committed in `weights/ai_screening/`.

#### Model versioning

Versions follow the format `v{code}r{revision}` (e.g. `v2r3`):
- **Code version** (`MODEL_CODE_VERSION` in `ai_classifier.py`): bump when architecture or feature extraction changes
- **Revision**: auto-incremented on each training run

Weights are saved to `weights/ai_screening/{version}.pt` and should be committed to the repo so the model is always available without retraining.

#### Training workflow

```bash
# 1. Run the DB migration first (adds ai_screening_* columns)
#    This runs automatically on API startup, or manually:
#    psql $DATABASE_URL -f pipeline/db/migrations/002_add_ai_screening.sql

# 2. Pre-compute DINOv2 features for all labelled videos
#    This is the slow step (~4s/video, network-bound). Resumable — skips cached.
#    Features are cached in data/ai_screening_cache/ (not committed).
python -m pipeline ai-extract --device mps  # use 'cuda' on Linux, 'cpu' as fallback

# 3. Train the classifier head (fast, ~30s — operates on cached features)
#    Saves weights to both data/ (ephemeral) and weights/ai_screening/ (committed)
python -m pipeline ai-train --epochs 50

# 4. Evaluate and calibrate the confidence threshold
python -m pipeline ai-eval --target-precision 0.95

# 5. Verify in the dev-tools UI
#    Models > AI Screening > "Sample & Inspect" — runs model on 20 random
#    labeled videos and shows accuracy, model vs human label per video,
#    title score, and vertical detection

# 6. Run AI screening on unscreened videos
python -m pipeline ai-screen --limit 10 --threshold 0.90

# 7. Commit the new weights
git add weights/ai_screening/
git commit -m "ai-screening: train v2r3 (95% accuracy on N samples)"
```

#### Running in Docker

```bash
# Install transformers (needed for DINOv2, not in base image)
docker exec argus-dev-api pip install transformers -q

# Extract, train, evaluate
docker exec -it argus-dev-api python3 -m pipeline.cli ai-extract --device cpu
docker exec -it argus-dev-api python3 -m pipeline.cli ai-train --epochs 50 --device cpu
docker exec -it argus-dev-api python3 -m pipeline.cli ai-eval
```

### Integration with Dev Tools

| Feature | Where to verify in dev-tools |
|---------|------------------------------|
| **Hard cuts** | `/videos/VIDEO_ID` → run "Detect Moves" → check segment count and per-move confidence in the move list |
| **Auto-calibration** | `POST /api/calibration/{channel}/propose` → compare returned bboxes with manual calibration. Or use the Calibration Editor page to view/edit proposed values |
| **AI screening** | Models > AI Screening > "Sample & Inspect" — shows model prediction vs human label per video with accuracy summary. Also available per-video at `/videos/VIDEO_ID` > Info > "Run AI Screen" |

---

## How to Add a New Dev Tool

The dev tools follow a **router -> service -> pipeline module** pattern. Every web tool should have a CLI equivalent.

1. **Service** (`dev-tools/api/services/<name>_service.py`): Business logic wrapping `pipeline.*` modules
2. **Router** (`dev-tools/api/routers/<name>.py`): FastAPI endpoints calling the service
3. **Register** in `dev-tools/api/main.py`: `app.include_router()`
4. **Page** (`dev-tools/app/<name>/page.tsx`): React page consuming the API
5. **Link** from dashboard (`dev-tools/app/page.tsx`)
6. **CLI parity**: Add or verify a matching CLI command in `pipeline/cli.py`

Reusable UI components live in `dev-tools/components/` — check `BboxDrawer`, `ChessBoard`, `MoveList`, and `FileUpload` before building new ones.
