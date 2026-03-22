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
