.PHONY: install dev test lint typecheck format train eval datagen infer clean \
       db-up db-down pipeline-install seed-channels crawl screen inspect download generate-clips pipeline-stats \
       dev-tools dev-tools-down blender-server blender-server-stop \
       up down

# ── Use venv Python/pip so targets work without activation ──

PYTHON := .venv/bin/python3
PIP := .venv/bin/pip

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

typecheck:
	$(PYTHON) -m mypy src/argus/

format:
	$(PYTHON) -m ruff format src/ tests/ scripts/
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/

train:
	$(PYTHON) scripts/train.py $(ARGS)

eval:
	$(PYTHON) scripts/evaluate.py $(ARGS)

datagen:
	$(PYTHON) scripts/generate_data.py $(ARGS)

infer:
	$(PYTHON) scripts/infer.py $(ARGS)

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Pipeline targets ─────────────────────────────────────────

db-up:
	docker compose up -d

db-down:
	docker compose down

pipeline-install:
	$(PIP) install -r pipeline/requirements.txt

seed-channels:
	$(PYTHON) -m pipeline.cli seed-channels

crawl:
	$(PYTHON) -m pipeline.cli crawl $(ARGS)

screen:
	$(PYTHON) -m pipeline.cli screen $(ARGS)

inspect:
	$(PYTHON) -m pipeline.cli inspect $(ARGS)

download:
	$(PYTHON) -m pipeline.cli download $(ARGS)

generate-clips:
	$(PYTHON) -m pipeline.cli generate-clips $(ARGS)

pipeline-stats:
	$(PYTHON) -m pipeline.cli stats

# ── Dev tools targets ────────────────────────────────────────

dev-tools:
	docker compose --profile dev-tools up --build

dev-tools-down:
	docker compose --profile dev-tools down

# ── Unified dev environment ─────────────────────────────────

BLENDER_PID_FILE := .blender-server.pid
BLENDER_LOG_FILE := .blender-server.log

up:
	@echo "Starting Docker services (postgres, dev-tools-api, dev-tools-ui)..."
	@docker compose --profile dev-tools up -d --build
	@echo ""
	@if command -v $(BLENDER) >/dev/null 2>&1 || [ -x "$(BLENDER)" ]; then \
		if lsof -ti tcp:$(BLENDER_PORT) >/dev/null 2>&1; then \
			echo "Blender server already running on port $(BLENDER_PORT)"; \
		else \
			echo "Starting Blender render server on port $(BLENDER_PORT)..."; \
			nohup $(BLENDER) --background --python blender/render_server.py -- \
				--port $(BLENDER_PORT) --quality training \
				> $(BLENDER_LOG_FILE) 2>&1 & \
			echo $$! > $(BLENDER_PID_FILE); \
		fi \
	else \
		echo "Blender not found — skipping render server (install Blender 4.0+ for synthetic data)"; \
	fi
	@echo ""
	@echo "=== All services up ==="
	@echo "  PostgreSQL:  localhost:$${POSTGRES_PORT:-5433}"
	@echo "  API:         http://localhost:8000"
	@echo "  UI:          http://localhost:3000"
	@echo "  Blender log: $(BLENDER_LOG_FILE)"
	@echo ""
	@echo "Stop with: make down"

down:
	@echo "Stopping Docker services..."
	@docker compose --profile dev-tools down
	@if [ -f $(BLENDER_PID_FILE) ]; then \
		kill $$(cat $(BLENDER_PID_FILE)) 2>/dev/null && echo "Blender server stopped" || true; \
		rm -f $(BLENDER_PID_FILE); \
	fi
	@lsof -ti tcp:$(BLENDER_PORT) | xargs kill 2>/dev/null || true
	@echo "All services stopped."

# ── Blender render server ───────────────────────────────────

BLENDER ?= $(shell which blender 2>/dev/null || echo "/Applications/Blender.app/Contents/MacOS/Blender")
BLENDER_PORT ?= 9876

blender-server:
	$(BLENDER) --background --python blender/render_server.py -- \
		--port $(BLENDER_PORT) --quality training $(ARGS)

blender-server-stop:
	@lsof -ti tcp:$(BLENDER_PORT) | xargs kill 2>/dev/null && echo "Blender server stopped" || echo "No server running on port $(BLENDER_PORT)"
