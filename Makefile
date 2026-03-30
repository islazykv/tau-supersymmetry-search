.PHONY: setup preprocess feature-engineer eda train train-bdt train-dnn tune \
       evaluate evaluate-bdt evaluate-dnn regions pipeline repro \
       test lint typecheck format clean ui dvc-push dvc-pull \
       docker-build docker-run serve serve-dnn

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

setup:
	uv sync
	pre-commit install

# --------------------------------------------------------------------------- #
# Pipeline stages (unified entry point)
# --------------------------------------------------------------------------- #

preprocess:
	uv run python run.py stage=preprocess

feature-engineer:
	uv run python run.py stage=feature_engineer

eda:
	uv run python run.py stage=eda

tune:
	uv run python run.py stage=tune

train: train-bdt

train-bdt:
	uv run python run.py stage=train

train-dnn:
	uv run python run.py stage=train model=dnn

evaluate: evaluate-bdt

evaluate-bdt:
	uv run python run.py stage=evaluate

evaluate-dnn:
	uv run python run.py stage=evaluate model=dnn

regions:
	uv run python run.py stage=regions

pipeline: preprocess feature-engineer train evaluate

repro:
	uv run dvc repro

# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/

format:
	uv run pre-commit run --all-files

# --------------------------------------------------------------------------- #
# Data versioning
# --------------------------------------------------------------------------- #

dvc-push:
	uv run dvc push

dvc-pull:
	uv run dvc pull

# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #

ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ outputs/

# --------------------------------------------------------------------------- #
# Docker
# --------------------------------------------------------------------------- #

docker-build:
	docker build -t tau-supersymmetry-search .

docker-run:
	docker run --rm tau-supersymmetry-search

# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #

serve:
	uv run python run.py stage=serve

serve-dnn:
	uv run python run.py stage=serve model=dnn
