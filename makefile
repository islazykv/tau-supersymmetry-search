.PHONY: setup preprocess feature-engineer eda train repro pipeline test format lint clean ui dvc-push dvc-pull

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup:
	uv sync
	pre-commit install

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

preprocess:
	uv run python preprocess.py

feature-engineer:
	uv run python feature_engineer.py

eda:
	uv run python eda.py

train:
	uv run python train.py

repro:
	uv run dvc repro

pipeline: preprocess feature-engineer train

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

format:
	pre-commit run --all-files

# ---------------------------------------------------------------------------
# Data versioning
# ---------------------------------------------------------------------------

dvc-push:
	uv run dvc push

dvc-pull:
	uv run dvc pull

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf outputs/
