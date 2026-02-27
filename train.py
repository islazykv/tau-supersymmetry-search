import logging

import hydra
import mlflow
import pyrootutils
from omegaconf import DictConfig, OmegaConf

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Run the training loop with MLflow experiment tracking."""
    log.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(cfg.get("experiment_name", "tau-supersymmetry-default"))

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        log.info("Training complete")


if __name__ == "__main__":
    main()
