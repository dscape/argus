.PHONY: install dev test lint typecheck format train eval datagen infer clean

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
