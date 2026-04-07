.PHONY: install dev test lint typecheck format train eval datagen infer train-pieces clean \
       db-up db-down db-backup db-restore pipeline-install seed-channels crawl screen inspect download generate-clips pipeline-stats \
       dev-tools dev-tools-down blender-server blender-server-stop \
       up down preview \
       docker-ai-extract docker-ai-train docker-ai-eval docker-ai-screen docker-ai-retrain \
       docker-ai-extract-status docker-smoke-test \
       backup check-backup \
       download-models

# ── Use venv Python/pip so targets work without activation ──

PYTHON := .venv/bin/python3
PIP := .venv/bin/pip

install:
	$(PIP) install -e .

dev: check-backup download-models
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

typecheck:
	$(PYTHON) -m mypy src/argus/
	cd dev-tools && npx tsc --noEmit

format:
	$(PYTHON) -m ruff format src/ tests/ scripts/
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/

train: check-backup
	$(PYTHON) scripts/train.py $(ARGS)

eval: check-backup
	$(PYTHON) scripts/evaluate.py $(ARGS)

datagen:
	$(PYTHON) scripts/generate_data.py $(ARGS)

infer:
	$(PYTHON) scripts/infer.py $(ARGS)

train-pieces: check-backup
	$(PYTHON) scripts/train_piece_classifier.py $(ARGS)

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Pipeline targets ─────────────────────────────────────────

db-up: check-backup
	docker compose up -d

db-down:
	@echo "Stopping database (data is preserved in Docker volume)."
	@echo "WARNING: 'docker compose down --volumes' will PERMANENTLY DELETE all data."
	docker compose down

backup:
	@scripts/backup.sh

check-backup:
	@scripts/check_backup.sh

db-backup:
	@mkdir -p backups
	docker compose exec -T postgres pg_dump -U argus argus > backups/argus_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup saved to backups/"
	@ls -lh backups/argus_*.sql | tail -1

db-restore:
	@if [ -z "$(BACKUP)" ]; then echo "Usage: make db-restore BACKUP=backups/argus_YYYYMMDD_HHMMSS.sql"; exit 1; fi
	@echo "Restoring from $(BACKUP)..."
	docker compose exec -T postgres psql -U argus argus < $(BACKUP)
	@echo "Restore complete."

pipeline-install:
	$(PIP) install -r pipeline/requirements.txt

seed-channels:
	$(PYTHON) -m pipeline.cli seed-channels

crawl:
	$(PYTHON) -m pipeline.cli crawl $(ARGS)

screen: check-backup
	$(PYTHON) -m pipeline.cli screen $(ARGS)

inspect:
	$(PYTHON) -m pipeline.cli inspect $(ARGS)

download:
	$(PYTHON) -m pipeline.cli download $(ARGS)

download-models:
	$(PYTHON) scripts/download_model.py

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

up: check-backup
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
		echo "Blender not found — skipping render server."; \
		echo "  Install:  brew install --cask blender   (macOS)"; \
		echo "            sudo snap install blender --classic   (Linux)"; \
		echo "  Docs:     See CONTRIBUTING.md § Prerequisites"; \
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

preview: check-backup
	@$(MAKE) down 2>/dev/null || true
	@if command -v $(BLENDER) >/dev/null 2>&1 || [ -x "$(BLENDER)" ]; then \
		echo "Starting Blender render server on port $(BLENDER_PORT)..."; \
		nohup $(BLENDER) --background --python blender/render_server.py -- \
			--port $(BLENDER_PORT) --quality training \
			> $(BLENDER_LOG_FILE) 2>&1 & \
		echo $$! > $(BLENDER_PID_FILE); \
	else \
		echo "Blender not found — skipping render server."; \
		echo "  Install:  brew install --cask blender   (macOS)"; \
		echo "            sudo snap install blender --classic   (Linux)"; \
		echo "  Docs:     See CONTRIBUTING.md § Prerequisites"; \
	fi
	docker compose --profile dev-tools up --build

# ── Blender render server ───────────────────────────────────

BLENDER ?= $(shell which blender 2>/dev/null || echo "/Applications/Blender.app/Contents/MacOS/Blender")
BLENDER_PORT ?= 9876

blender-server:
	$(BLENDER) --background --python blender/render_server.py -- \
		--port $(BLENDER_PORT) --quality training $(ARGS)

blender-server-stop:
	@lsof -ti tcp:$(BLENDER_PORT) | xargs kill 2>/dev/null && echo "Blender server stopped" || echo "No server running on port $(BLENDER_PORT)"

# ── Docker-wrapped pipeline targets ────────────────────────

docker-ai-extract:
	docker exec -it argus-dev-api python3 -m pipeline.cli ai-extract $(ARGS)

docker-ai-train:
	docker exec -it argus-dev-api python3 -m pipeline.cli ai-train $(ARGS)

docker-ai-eval:
	docker exec -it argus-dev-api python3 -m pipeline.cli ai-eval $(ARGS)

docker-ai-screen:
	docker exec -it argus-dev-api python3 -m pipeline.cli ai-screen $(ARGS)

docker-ai-retrain:
	docker exec -it argus-dev-api python3 -m pipeline.cli ai-retrain $(ARGS)

docker-ai-extract-status:
	docker exec argus-dev-api python3 -m pipeline.cli ai-extract-status

docker-smoke-test:
	docker exec argus-dev-api python3 -m pipeline.cli smoke-test
