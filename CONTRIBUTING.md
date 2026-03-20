# Contributing to Argus

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | ML code and pipeline |
| Node.js | 18+ | Dev tools web UI |
| Docker | Latest | PostgreSQL database |
| Cairo | Latest | SVG rendering for synthetic data |
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

## Development Setup

```bash
# 1. Clone and create virtualenv
git clone <repo-url> && cd argus
python3 -m venv .venv
source .venv/bin/activate

# 2. Install ML code + dev dependencies
make dev

# 3. Install pipeline dependencies (separate from ML deps)
make pipeline-install

# 4. Start PostgreSQL and apply schema
make db-up

# 5. Copy and fill in environment variables
cp .env.example .env
# Edit .env: DATABASE_URL, YOUTUBE_API_KEY, ANTHROPIC_API_KEY

# 6. (Optional) Start the dev tools web UI
cd dev-tools && npm install && npm run dev   # Frontend: localhost:3000
cd dev-tools/api && uvicorn main:app --reload --port 8000  # Backend: localhost:8000
```

## Running Tests & Checks

```bash
make test       # pytest (183 tests)
make lint       # ruff check
make typecheck  # mypy
make format     # ruff format + auto-fix
```

## Project Layout

The codebase has three mostly independent packages:

```
src/argus/    ML model, training, inference, evaluation, data generation
pipeline/     Data pipeline (crawl, extract, match, download, clips, overlay)
dev-tools/    Next.js web UI + FastAPI backend for developer inspection tools
```

`src/argus/` and `pipeline/` share only the `argus.chess` module (for move vocabulary and PGN verification). They have separate dependency sets — see `pyproject.toml` vs `pipeline/requirements.txt`.

`dev-tools/` wraps `pipeline.overlay.*` modules behind a REST API.

## Code Style

- **Formatter/linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Line length**: 100
- **Target**: Python 3.10
- **Rules**: E, F, I (isort), N (naming), W, UP (pyupgrade)
- **Type checking**: mypy strict mode

Run `make format` before committing.

## How to Add a New Pipeline Stage

1. Create a module under `pipeline/<stage_name>/`
2. Add the main function (e.g., `run_all()`)
3. Register the CLI command in `pipeline/cli.py`:
   - Add a `cmd_<name>(args)` function
   - Add the subparser in `main()`
   - Add the dispatch entry in the `commands` dict
4. Add a Makefile target if it's a standard pipeline step
5. Add tests under `tests/pipeline/`

## How to Add a New Dev Tool

The dev tools follow a router → service → pipeline pattern:

1. **Service** (`dev-tools/api/services/<name>_service.py`): Business logic wrapping `pipeline.*` modules
2. **Router** (`dev-tools/api/routers/<name>.py`): FastAPI endpoints calling the service
3. **Register** in `dev-tools/api/main.py`: `app.include_router()`
4. **Page** (`dev-tools/app/<name>/page.tsx`): React page consuming the API
5. **Link** from dashboard (`dev-tools/app/page.tsx`)

Reusable UI components live in `dev-tools/components/` — check `BboxDrawer`, `ChessBoard`, `MoveList`, and `FileUpload` before building new ones.

## Hydra Configuration

Training, data, and model configs live in `configs/`. Override any parameter from CLI:

```bash
make train ARGS="training.batch_size=16 training.optimizer.lr=5e-4"
```

When adding a new config group, create a YAML file in the appropriate subdirectory and reference it from `configs/config.yaml`.
