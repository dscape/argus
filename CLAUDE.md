# Argus

Chess vision model using VLA (Vision-Language-Action) paradigm to detect moves from OTB (over-the-board) chess footage.

## Dev Environment

### Starting the dev servers

The dev UI requires **both** the FastAPI backend and the Next.js frontend. Always start both:

```bash
# Option 1: via launch.json (preferred in Claude Code)
# Start "api" and "dev-tools" servers from .claude/launch.json

# Option 2: via Make (uses Docker)
make dev-tools

# Option 3: manually
cd dev-tools
../.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload  # backend
npm run dev                                                               # frontend
```

The frontend (Next.js, port 3000) proxies `/api/*` to the backend (FastAPI, port 8000). If the backend is not running, pages will show "Loading..." and all API calls will fail with proxy errors.

### Key paths

- `dev-tools/` — Next.js frontend + FastAPI backend (co-located)
- `dev-tools/api/` — FastAPI routers and services
- `dev-tools/app/` — Next.js app router pages
- `dev-tools/lib/` — Shared TypeScript utilities and API client
- `pipeline/` — Python ML pipeline (overlay reading, move detection, calibration, data generation)
- `src/argus/` — Core model and training code
- `pipeline/db/` — Database schema and migrations (PostgreSQL)

### Type checking

```bash
make typecheck   # runs mypy + tsc
```

### Testing

```bash
make test        # pytest
```
