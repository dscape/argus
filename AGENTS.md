# LLM / Agent Instructions for Argus

## CLI-First Development

When running pipeline operations, **always** use the CLI:

```
python -m pipeline <command> [options]
```

Available commands: `db-init`, `seed-channels`, `resolve-channels`, `crawl`, `screen`, `inspect`, `download`, `calibrate`, `generate-clips`, `overlay-test`, `overlay-test-reader`, `inspect-clip`, `ai-extract`, `ai-train`, `ai-eval`, `ai-screen`, `auto-calibrate`, `smoke-test`, `inspect-calibration`, `ai-extract-status`, `stats`.

**Do NOT** write one-off `python -c "..."` inline scripts. If a command doesn't exist for what you need, add a new subcommand to `pipeline/cli.py` following the existing pattern:

1. Add a `cmd_<name>(args)` function with lazy imports
2. Add a subparser in `main()`
3. Add the dispatch entry in the `commands` dict

## Docker Execution

When running inside Docker, use Makefile targets:

```
make docker-ai-extract ARGS="--device cpu"
make docker-ai-train ARGS="--epochs 50"
make docker-ai-eval
make docker-ai-extract-status
make docker-smoke-test
```

Or directly: `docker exec argus-dev-api python3 -m pipeline.cli <command>`.

## Dev Tools Architecture

The dev-tools web UI follows a strict layered architecture:

```
Router  (dev-tools/api/routers/*.py)      — FastAPI endpoints
  -> Service  (dev-tools/api/services/*_service.py) — Business logic
    -> Pipeline module  (pipeline/**/*.py)           — Core functionality
```

When adding a new dev tool feature:

1. Add business logic in a service file under `dev-tools/api/services/`
2. Create FastAPI endpoints in a router under `dev-tools/api/routers/`
3. Register the router in `dev-tools/api/main.py`
4. Add TypeScript API functions in `dev-tools/lib/api.ts`
5. Add TypeScript types in `dev-tools/lib/types.ts`
6. Build the UI page in `dev-tools/app/<name>/page.tsx`
7. Ensure a matching CLI command exists in `pipeline/cli.py`

Reusable UI components are in `dev-tools/components/` — check before building new ones.

## Background Jobs

For long-running tasks (generation, extraction), use the in-memory job pattern:
- Thread + job dict (see `dev-tools/api/services/generation_service.py`)
- Poll-based status endpoint
- Cancel via `threading.Event`

## Key Conventions

- Python 3.10+, Ruff formatter (100 char lines), mypy strict mode
- React/Next.js
- PostgreSQL for persistence, in-memory dicts for ephemeral job state
- Run `make format` before committing
- **Validation**: Before completing any task that modifies code, always run `make typecheck` (covers both Python/mypy and TypeScript/tsc), `make lint`, and `make test` — both must pass with 0 errors
- When you do changes to dev tools web app use `preview` or `browser` functionality to visually inspect the desired result
- When you train a model, or you change the source code of a model you need to change the version. v2 if it's the second version, r10 if it is the 10th training run of said model version
