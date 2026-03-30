"""Model serving pipeline: start the FastAPI inference server."""

from __future__ import annotations

import logging

import pyrootutils
import uvicorn
from omegaconf import DictConfig

from src.processing.analysis import get_output_paths
from src.serving.app import create_app

log = logging.getLogger(__name__)

_MODEL_FILE = {
    "xgboost": ("bdt", "bdt.ubj"),
    "pytorch_dnn": ("dnn", "dnn.pt"),
}


def serve(cfg: DictConfig) -> None:
    """Start the inference server for the configured model type."""
    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    models_dir = root / output_paths["models_dir"]

    model_name = cfg.model.name
    if model_name not in _MODEL_FILE:
        raise ValueError(f"Unknown model for serving: {model_name!r}")

    model_type, filename = _MODEL_FILE[model_name]
    model_path = models_dir / filename

    log.info("Starting %s server — model: %s", model_type.upper(), model_path)

    app = create_app(model_type=model_type, model_path=str(model_path))
    uvicorn.run(app, host="0.0.0.0", port=8000)
