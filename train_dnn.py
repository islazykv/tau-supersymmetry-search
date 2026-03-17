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

from src.models.dnn import (  # noqa: E402
    build_criterion,
    build_model,
    build_scaler,
    predict,
    resolve_device,
    save_model,
    train,
    train_kfold,
)
from src.models.plots import plot_dnn_kfold_training_curves, plot_dnn_training_curves  # noqa: E402
from src.models.splits import (  # noqa: E402
    build_predictions_frame,
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
    """Run the full DNN training pipeline with MLflow experiment tracking."""
    # Override model to DNN if not already set
    if cfg.model.name != "pytorch_dnn":
        raise ValueError(
            f"Expected model=dnn config, got model.name={cfg.model.name!r}. "
            "Run with: python train_dnn.py model=dnn"
        )

    log.info("Starting DNN training:\n%s", OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(f"file://{root}/mlruns")
    mlflow.set_experiment(cfg.get("experiment_name", "tau-supersymmetry-search-dnn"))

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

            # --- resolve paths ---
            output_paths = get_output_paths(cfg)
            dataframes_dir = root / output_paths["dataframes_dir"]
            models_dir = root / output_paths["models_dir"]
            plots_dir = root / output_paths["plots_dir"] / "dnn"
            models_dir.mkdir(parents=True, exist_ok=True)
            plots_dir.mkdir(parents=True, exist_ok=True)

            # --- load data ---
            df_mc = load_dataframe(dataframes_dir / "mc.parquet")
            log.info("Loaded MC: %d events, %d columns", len(df_mc), len(df_mc.columns))

            # --- class labels ---
            class_names = get_class_names(df_mc)
            n_classes = len(class_names)
            log.info("Classes (%d): %s", n_classes, class_names)
            mlflow.log_param("n_classes", n_classes)
            mlflow.log_param("class_names", class_names)

            # --- features & target ---
            X, y, weights = prepare_features_target(df_mc)
            mlflow.log_param("n_features", X.shape[1])

            # --- split ---
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

            # --- device ---
            device = resolve_device()
            log.info("Device: %s", device)

            # --- train ---
            if split_strategy == "train_test":
                model = build_model(cfg, n_features=X.shape[1], n_classes=n_classes)
                model = model.to(device)

                scaler = build_scaler(X_train)

                class_weight_per_class = (
                    w_train.groupby(y_train).first().sort_index().values
                )
                criterion = build_criterion(class_weight_per_class, device)

                model, history = train(
                    model,
                    criterion,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    scaler=scaler,
                    cfg=cfg,
                    device=device,
                    early_stopping_rounds=cfg.pipeline.early_stopping_rounds,
                )

                mlflow.log_metric("best_epoch", history["best_epoch"])
                mlflow.log_metric(
                    "best_val_loss", history["val_loss"][history["best_epoch"]]
                )

                for step, (t_loss, v_loss) in enumerate(
                    zip(history["train_loss"], history["val_loss"])
                ):
                    mlflow.log_metric("train_loss", t_loss, step=step)
                    mlflow.log_metric("val_loss", v_loss, step=step)

                # training curve plot
                fig = plot_dnn_training_curves(history)
                curve_path = plots_dir / "training_curves.png"
                save_figure(fig, curve_path)
                mlflow.log_artifact(str(curve_path))

                # predict
                y_pred, y_proba = predict(
                    model, X_test, scaler, device, batch_size=cfg.model.batch_size
                )

            elif split_strategy == "k_fold":
                models, scalers, y_pred, y_proba, y_test, fold_histories = train_kfold(
                    cfg,
                    folds,
                    n_classes=n_classes,
                    device=device,
                    early_stopping_rounds=cfg.pipeline.early_stopping_rounds,
                )
                for fold_idx, h in enumerate(fold_histories):
                    mlflow.log_metric("best_epoch", h["best_epoch"], step=fold_idx)
                    mlflow.log_metric(
                        "best_val_loss", h["val_loss"][h["best_epoch"]], step=fold_idx
                    )

                # k-fold training curve plot
                fig = plot_dnn_kfold_training_curves(fold_histories)
                curve_path = plots_dir / "training_curves_kfold.png"
                save_figure(fig, curve_path)
                mlflow.log_artifact(str(curve_path))

            # --- predictions DataFrame ---
            predictions_df = build_predictions_frame(
                y_test, y_pred, y_proba, class_names
            )
            predictions_path = dataframes_dir / "dnn_predictions.parquet"
            save_dataframe(predictions_df, predictions_path)
            mlflow.log_artifact(str(predictions_path))

            # --- save model(s) ---
            if cfg.pipeline.save_model:
                if split_strategy == "train_test":
                    model_path = models_dir / "dnn.pt"
                    save_model(model, scaler, model_path)
                    mlflow.log_artifact(str(model_path))
                elif split_strategy == "k_fold":
                    for fold_idx, (m, s) in enumerate(zip(models, scalers)):
                        model_path = models_dir / f"dnn_fold{fold_idx}.pt"
                        save_model(m, s, model_path)
                        mlflow.log_artifact(str(model_path))

            log.info("Training complete — predictions and model saved")
        except Exception:
            mlflow.set_tag("mlflow.runStatus", "FAILED")
            log.exception("Training failed")
            raise


if __name__ == "__main__":
    main()
