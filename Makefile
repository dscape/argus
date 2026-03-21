.PHONY: install dev test lint typecheck format train eval datagen infer clean \
       db-up db-down pipeline-install import-data crawl extract match download-videos generate-clips pipeline-stats \
       dev-tools dev-tools-down blender-server blender-server-stop \
       up down

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/

typecheck:
	mypy src/argus/

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

train:
	python3 scripts/train.py $(ARGS)

eval:
	python3 scripts/evaluate.py $(ARGS)

datagen:
	python3 scripts/generate_data.py $(ARGS)

infer:
	python3 scripts/infer.py $(ARGS)

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Pipeline targets ─────────────────────────────────────────

db-up:
	docker compose up -d

db-down:
	docker compose down

pipeline-install:
	pip install -r pipeline/requirements.txt

import-data:
	python3 -m pipeline.cli import-players
	python3 -m pipeline.cli import-pgns $(ARGS)
	python3 -m pipeline.cli seed-channels

crawl:
	python3 -m pipeline.cli crawl $(ARGS)

extract:
	python3 -m pipeline.cli extract $(ARGS)

match:
	python3 -m pipeline.cli match $(ARGS)

download-videos:
	python3 -m pipeline.cli download $(ARGS)

generate-clips:
	python3 -m pipeline.cli generate-clips $(ARGS)

pipeline-stats:
	python3 -m pipeline.cli stats

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
