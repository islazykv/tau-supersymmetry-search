import logging

import hydra
import matplotlib
import mlflow
import pyrootutils
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")

logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

from src.models.bdt import (  # noqa: E402
    build_params,
    build_predictions_frame,
    get_evals_result,
    predict,
    save_model,
    train,
    train_kfold,
)
from src.models.plots import plot_kfold_training_curves, plot_training_curves  # noqa: E402
from src.models.splits import (  # noqa: E402
    kfold_split,
    prepare_features_target,
    train_test_split,
)
from src.eda.utils import get_class_names  # noqa: E402
from src.processing.analysis import get_output_paths  # noqa: E402
from src.processing.io import load_dataframe, save_dataframe  # noqa: E402
from src.visualization.plots import save_figure  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the full BDT training pipeline with MLflow experiment tracking."""
    log.info("Starting BDT training:\n%s", OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(f"file://{root}/mlruns")
    mlflow.set_experiment(cfg.get("experiment_name", "tau-supersymmetry-search-bdt"))

    with mlflow.start_run():
        try:
            mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
            mlflow.log_params(
                {
                    "split_strategy": cfg.pipeline.split_strategy,
                    "n_splits": cfg.pipeline.n_splits,
                    "early_stopping_rounds": cfg.pipeline.early_stopping_rounds,
                    "test_split": cfg.data.test_split,
                    "seed": cfg.seed,
                }
            )

            output_paths = get_output_paths(cfg)
            dataframes_dir = root / output_paths["dataframes_dir"]
            models_dir = root / output_paths["models_dir"]
            plots_dir = root / output_paths["plots_dir"] / "bdt"
            models_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            df_mc = load_dataframe(dataframes_dir / "mc.parquet")
            log.info("Loaded MC: %d events, %d columns", len(df_mc), len(df_mc.columns))

            class_names = get_class_names(df_mc)
            n_classes = len(class_names)
            log.info("Classes (%d): %s", n_classes, class_names)
            mlflow.log_param("n_classes", n_classes)
            mlflow.log_param("class_names", class_names)

            X, y, weights = prepare_features_target(df_mc)
            mlflow.log_param("n_features", X.shape[1])

            split_strategy = cfg.pipeline.split_strategy

            if split_strategy == "train_test":
                X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
                    X,
                    y,
                    weights,
                    test_size=cfg.data.test_split,
                    seed=cfg.seed,
                )
                log.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

            elif split_strategy == "k_fold":
                folds = kfold_split(
                    X, y, weights, n_splits=cfg.pipeline.n_splits, seed=cfg.seed
                )
                log.info("K-fold: %d stratified folds", cfg.pipeline.n_splits)

            else:
                raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

            params = build_params(cfg, n_classes=n_classes)
            metric = params["eval_metric"]

            if split_strategy == "train_test":
                model = train(
                    params,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    w_train=w_train,
                    early_stopping_rounds=cfg.pipeline.early_stopping_rounds,
                )
                mlflow.log_metric("best_iteration", model.best_iteration)
                mlflow.log_metric(f"best_val_{metric}", model.best_score)

                # log per-round metrics
                evals = get_evals_result(model)
                for step, (train_val, val_val) in enumerate(
                    zip(
                        evals["validation_0"][metric],
                        evals["validation_1"][metric],
                    )
                ):
                    mlflow.log_metric(f"train_{metric}", train_val, step=step)
                    mlflow.log_metric(f"val_{metric}", val_val, step=step)

                # training curve plot
                fig = plot_training_curves(evals, metric=metric)
                curve_path = plots_dir / "training_curves.png"
                save_figure(fig, curve_path)
                mlflow.log_artifact(str(curve_path))

                # predict
                y_pred, y_proba = predict(model, X_test)

            elif split_strategy == "k_fold":
                models, y_pred, y_proba, y_test = train_kfold(
                    params,
                    folds,
                    early_stopping_rounds=cfg.pipeline.early_stopping_rounds,
                )
                for fold_idx, m in enumerate(models):
                    mlflow.log_metric("best_iteration", m.best_iteration, step=fold_idx)
                    mlflow.log_metric(f"best_val_{metric}", m.best_score, step=fold_idx)

                # k-fold training curve plot
                fig = plot_kfold_training_curves(models, metric=metric)
                curve_path = plots_dir / "training_curves_kfold.png"
                save_figure(fig, curve_path)
                mlflow.log_artifact(str(curve_path))

            predictions_df = build_predictions_frame(
                y_test, y_pred, y_proba, class_names
            )
            predictions_path = dataframes_dir / "bdt_predictions.parquet"
            save_dataframe(predictions_df, predictions_path)
            mlflow.log_artifact(str(predictions_path))

            if cfg.pipeline.save_model:
                if split_strategy == "train_test":
                    model_path = models_dir / "bdt.ubj"
                    save_model(model, model_path)
                    mlflow.log_artifact(str(model_path))
                elif split_strategy == "k_fold":
                    for fold_idx, m in enumerate(models):
                        model_path = models_dir / f"bdt_fold{fold_idx}.ubj"
                        save_model(m, model_path)
                        mlflow.log_artifact(str(model_path))

            log.info("Training complete — predictions and model saved")
        except Exception:
            mlflow.set_tag("mlflow.runStatus", "FAILED")
            log.exception("Training failed")
            raise


if __name__ == "__main__":
    main()
