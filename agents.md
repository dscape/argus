# Argus

Chess vision model using VLA (Vision-Language-Action) paradigm to detect moves from OTB (over-the-board) chess footage.

## Dev Environment

```bash
make up          # starts postgres, API, UI, and Blender render server
make down        # stops everything
make typecheck   # mypy + tsc
make test        # pytest
make lint        # ruff check
make format      # ruff format + auto-fix
```

`make up`, `make test`, and runtime pipeline targets preflight committed model weights under `weights/`. After cloning, install Git LFS and run `git lfs pull`.

UI: http://localhost:3000 | API: http://localhost:8000. Frontend proxies `/api/*` to backend.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/argus/` | Core model and training code (VLA architecture) |
| `pipeline/` | ML pipeline: overlay reading, screening, calibration |
| `dev-tools/` | Next.js frontend + FastAPI backend (co-located) |
| `configs/` | Hydra configs: model, training, data, evaluate, datagen |
| `data/` | Training data, cached features, board images |
| `weights/` | Committed model weights (screening, overlay_yolo, overlay piece-classifier, dinov2-base) |
| `pipeline/db/` | PostgreSQL schema and migrations |
| `outputs/` | For any sort of non-transient outputs, for human inspection |
| `scripts/` | Scripts that can be re-used to assist development of argus |

## Naming Conventions

Consistent identifiers used across nav, code, data dirs, configs, and weights:

| Identifier | Meaning |
|-----------|---------|
| `screening` | Video classification (OTB vs not) |
| `overlay` | Overlay pipeline in general: runtime localization, bbox training labels, piece classification |
| `segmentation` | Video segmentation |
| `calibration` | Board calibration |
| `argus` | The main VLA model |

Data splits: `train/`, `val/`, `val_real/`. Model weights: `weights/screening/`, `weights/overlay/` (piece classifier), `weights/overlay_yolo/` (default runtime overlay detector). Cached features: `dataset/torch/`. Raw frames: `dataset/frames/`.

## Dev Tools Architecture

Four sections matching the data pipeline: **Videos** > **Annotate** > **Data** > **Evaluate**

```
Router  (dev-tools/api/routers/{videos,annotate,data,evaluate}/)
  -> Service  (dev-tools/api/services/{videos,annotate,data,evaluate}/)
    -> Pipeline module  (pipeline/**/*.py)
```

## CLI-First Pipeline

Always use `python -m pipeline <command>` for pipeline operations. Do not write inline scripts. If a command doesn't exist, add a subcommand to `pipeline/cli.py`.

Docker: `docker exec argus-dev-api python3 -m pipeline.cli <command>` or use `make docker-*` targets.

## Progress Tracking

- Always read `progress.md` before continuing active branch work or resuming an interrupted task.
- Keep `progress.md` updated as findings, decisions, architecture changes, and validation results land.
- Before handing work back, review `progress.md` and make sure it reflects the current state of the branch.

## Conventions

- Python 3.10+, Ruff (100 char lines), mypy strict mode
- React/Next.js with shadcn/ui components
- PostgreSQL for persistence, in-memory dicts for ephemeral job state
- Background jobs: thread + job dict, poll-based status, cancel via `threading.Event`
- Model versioning: `v{code}r{revision}` (e.g. `v2r3`). Bump code version on architecture changes
- Default runtime overlay localization uses the committed YOLO detector in `weights/overlay_yolo/`
- `data/videos/ground_truth.json`, `tests/fixtures/frames/ground_truth.json`, and `/annotate/bbox` are detector training/eval labels only — runtime does not read those bboxes directly
- If you change the overlay detector, run `scripts/visualize_overlay_tests.py` on every iteration and treat that visual output as mandatory validation
- Before completing tasks: run `make typecheck`, `make lint`, `make test` — all must pass
