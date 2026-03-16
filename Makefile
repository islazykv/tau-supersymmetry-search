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
# Pipeline stages
# --------------------------------------------------------------------------- #

preprocess:
	uv run python preprocess.py

feature-engineer:
	uv run python feature_engineer.py

eda:
	uv run python eda.py

tune:
	uv run python tune.py

train: train-bdt

train-bdt:
	uv run python train_bdt.py

train-dnn:
	uv run python train_dnn.py

evaluate: evaluate-bdt

evaluate-bdt:
	uv run python evaluate_bdt.py

evaluate-dnn:
	uv run python evaluate_dnn.py

regions:
	uv run python regions.py

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
	docker build -t tau-supersymmetry .

docker-run:
	docker run --rm tau-supersymmetry

# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #

serve:
	uv run python serve.py --model-type bdt --model-path $(MODEL_PATH)

serve-dnn:
	uv run python serve.py --model-type dnn --model-path $(MODEL_PATH)
