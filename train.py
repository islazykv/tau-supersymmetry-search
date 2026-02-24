import pyrootutils
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):

    print(f"Starting training with config:\n{OmegaConf.to_yaml(cfg)}")

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(cfg.get("experiment_name", "tau-supersymmetry-default"))

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        print("Training complete!")


if __name__ == "__main__":
    main()
