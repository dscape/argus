.PHONY: install dev test lint typecheck format train eval datagen infer clean \
       db-up db-down pipeline-install import-data crawl extract match download-videos generate-clips pipeline-stats \
       dev-tools dev-tools-down

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
