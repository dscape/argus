# Contributing to Argus

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | ML code and pipeline |
| Node.js | 18+ | Dev tools web UI |
| Docker | Latest | PostgreSQL + dev tools |
| Cairo | Latest | SVG rendering for synthetic data |
| Blender | 4.0+ | 3D synthetic data rendering (optional) |
| ffmpeg | Latest | Frame extraction from videos |

### Install Cairo

```bash
# macOS
brew install cairo

# Ubuntu / Debian
sudo apt-get install libcairo2-dev

# Fedora
sudo dnf install cairo-devel
```

### Install Blender (optional, for 3D data generation)

```bash
# macOS
brew install --cask blender

# Ubuntu / Debian
sudo snap install blender --classic

# Or download from https://www.blender.org/download/
```

Verify: `blender --version` should show 4.0 or later. You can also set the `BLENDER_PATH` environment variable to point to a custom Blender install.

---

## Which Domain?

| I want to... | Domain | Folder | Setup needed |
|--------------|--------|--------|-------------|
| Add YouTube channels or improve game matching | Pipeline | `pipeline/` | Python + PostgreSQL + API keys |
| Improve overlay board reading or calibration | Pipeline (overlay) | `pipeline/overlay/` | Python + Cairo |
| Generate better synthetic training data | Data Generation | `src/argus/datagen/` | Python + Cairo (+ Blender 4.0+ for 3D) |
| Train the model or tune hyperparameters | Training | `src/argus/model/`, `src/argus/training/` | Python + PyTorch + GPU |
| Improve inference accuracy or add streaming | Inference | `src/argus/inference/` | Python + PyTorch + trained checkpoint |
| Build or improve developer inspection tools | Dev Tools | `dev-tools/` | Docker (or Python + Node.js) |

---

## Development Setup

### Common (all domains)

```bash
git clone <repo-url> && cd argus
python3 -m venv .venv
source .venv/bin/activate
make dev  # installs src/argus + dev dependencies (pytest, ruff, mypy)
```

### Pipeline

```bash
make pipeline-install   # installs pipeline/ dependencies (psycopg, yt-dlp, etc.)
make db-up              # starts PostgreSQL via docker-compose
cp .env.example .env    # fill in DATABASE_URL, YOUTUBE_API_KEY, ANTHROPIC_API_KEY
```

### Data Generation

No extra setup beyond `make dev`. Cairo must be installed system-wide (see Prerequisites).

For 3D data, install Blender 4.0+ (see Prerequisites). The STL piece models are included in `blender/models/staunton/`. Set `BLENDER_PATH` if Blender is not on your PATH.

### Training

```bash
# CPU/MPS (default — uses GRU fallback for temporal module)
make dev

# GPU with Mamba-2 (requires CUDA)
pip install -e ".[cuda,dev]"
```

### Dev Tools

```bash
# Option 1: Docker (recommended)
make dev-tools  # starts PostgreSQL + FastAPI + Next.js

# Option 2: Manual (for dev-tools development)
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
| **Pipeline — extraction** | `pytest tests/pipeline/test_title_parser.py tests/pipeline/test_description_parser.py tests/pipeline/test_player_aliases.py -v` | Title/description parsing, player name normalization |
| **Pipeline — matching** | `pytest tests/pipeline/test_scoring.py tests/pipeline/test_pgn_verifier.py tests/pipeline/test_pgn_aligner.py -v` | Match scoring, PGN verification, move alignment |
| **Pipeline — overlay** | `pytest tests/pipeline/test_overlay_reader.py tests/pipeline/test_overlay_move_detector.py -v` | Overlay FEN reading, move detection |

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

## How to Add a New Dev Tool

The dev tools follow a **router -> service -> pipeline module** pattern. Every web tool should have a CLI equivalent.

1. **Service** (`dev-tools/api/services/<name>_service.py`): Business logic wrapping `pipeline.*` modules
2. **Router** (`dev-tools/api/routers/<name>.py`): FastAPI endpoints calling the service
3. **Register** in `dev-tools/api/main.py`: `app.include_router()`
4. **Page** (`dev-tools/app/<name>/page.tsx`): React page consuming the API
5. **Link** from dashboard (`dev-tools/app/page.tsx`)
6. **CLI parity**: Add or verify a matching CLI command in `pipeline/cli.py`

Reusable UI components live in `dev-tools/components/` — check `BboxDrawer`, `ChessBoard`, `MoveList`, and `FileUpload` before building new ones.
