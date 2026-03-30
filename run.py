"""Unified entry point for the tau supersymmetry search pipeline.

Usage:
    uv run python run.py stage=preprocess
    uv run python run.py stage=feature_engineer
    uv run python run.py stage=eda
    uv run python run.py stage=train
    uv run python run.py stage=train model=dnn
    uv run python run.py stage=evaluate
    uv run python run.py stage=evaluate model=dnn
    uv run python run.py stage=tune
    uv run python run.py stage=tune model=dnn
    uv run python run.py stage=regions
    uv run python run.py stage=serve
    uv run python run.py stage=serve model=dnn
"""

from __future__ import annotations

import logging

import hydra
import matplotlib
import pyrootutils
from omegaconf import DictConfig

matplotlib.use("Agg")

logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Dispatch to the appropriate pipeline stage."""
    stage = cfg.stage
    log.info("Running stage: %s", stage)

    match stage:
        case "preprocess":
            from src.pipelines.preprocess import preprocess

            preprocess(cfg)

        case "feature_engineer":
            from src.pipelines.feature_engineer import feature_engineer

            feature_engineer(cfg)

        case "eda":
            from src.pipelines.eda import eda

            eda(cfg)

        case "train":
            model_name = cfg.model.name
            if model_name == "xgboost":
                from src.pipelines.train_bdt import train_bdt

                train_bdt(cfg)
            elif model_name == "pytorch_dnn":
                from src.pipelines.train_dnn import train_dnn

                train_dnn(cfg)
            else:
                raise ValueError(f"Unknown model for training: {model_name!r}")

        case "evaluate":
            model_name = cfg.model.name
            if model_name == "xgboost":
                from src.pipelines.evaluate_bdt import evaluate_bdt

                evaluate_bdt(cfg)
            elif model_name == "pytorch_dnn":
                from src.pipelines.evaluate_dnn import evaluate_dnn

                evaluate_dnn(cfg)
            else:
                raise ValueError(f"Unknown model for evaluation: {model_name!r}")

        case "tune":
            from src.pipelines.tune import tune

            tune(cfg)

        case "regions":
            from src.pipelines.regions import regions

            regions(cfg)

        case "serve":
            from src.pipelines.serve import serve

            serve(cfg)

        case _:
            msg = f"Unknown stage: {stage}"
            raise ValueError(msg)


if __name__ == "__main__":
    main()
