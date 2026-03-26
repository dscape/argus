# Argus

Chess vision model using VLA (Vision-Language-Action) paradigm to detect moves from OTB (over-the-board) chess footage.

## Dev Environment

### Starting the dev servers

Everything runs via Docker Compose. Start and stop with:

```bash
make up    # starts postgres, API, UI, and Blender render server
make down  # stops everything
```

The UI is at http://localhost:3000, the API at http://localhost:8000. The frontend proxies `/api/*` to the backend. If the backend is not running, pages will show "Loading..." and all API calls will fail with proxy errors.

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
