.PHONY: setup train ui format clean

setup:
	uv sync
	pre-commit install

train:
	uv run python train.py

ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

format:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
