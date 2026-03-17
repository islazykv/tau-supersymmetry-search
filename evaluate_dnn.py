import json
import logging
import re

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

from src.eda.utils import get_class_labels, get_class_names  # noqa: E402
from src.models.dnn import load_model, resolve_device  # noqa: E402
from src.models.evaluation import (  # noqa: E402
    compute_dnn_shap_values,
    compute_summary_metrics,
    plot_classification_report,
    plot_confusion_matrix,
    plot_permutation_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_score_distributions,
    plot_shap_importance,
)
from src.models.splits import prepare_features_target  # noqa: E402
from src.processing.analysis import get_output_paths  # noqa: E402
from src.processing.io import load_dataframe  # noqa: E402
from src.visualization.plots import save_figure  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the full DNN evaluation pipeline with MLflow experiment tracking."""

    log.info("Starting DNN evaluation:\n%s", OmegaConf.to_yaml(cfg))

    # --- mlflow: resume latest DNN training run ---
    tracking_uri = f"file://{root}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = cfg.get("experiment_name", "tau-supersymmetry-search-dnn")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    latest_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = latest_runs[0].info.run_id
    log.info("Resuming MLflow run: %s", run_id)

    with mlflow.start_run(run_id=run_id):
        # --- resolve paths ---
        output_paths = get_output_paths(cfg)
        dataframes_dir = root / output_paths["dataframes_dir"]
        models_dir = root / output_paths["models_dir"]
        plots_dir = root / output_paths["plots_dir"] / "dnn_evaluation"
        metrics_dir = root / output_paths["dataframes_dir"].parent / "metrics"
        plots_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # --- load data ---
        df_mc = load_dataframe(dataframes_dir / "mc.parquet")
        log.info("Loaded MC: %d events, %d columns", len(df_mc), len(df_mc.columns))

        # --- class labels ---
        display_labels = OmegaConf.to_container(cfg.merge.display_labels, resolve=True)
        class_names = get_class_names(df_mc)
        class_labels = get_class_labels(df_mc, display_labels=display_labels)
        log.info("Classes (%d): %s", len(class_names), class_names)

        # --- device ---
        device = resolve_device()

        # --- load model(s) ---
        split_strategy = cfg.pipeline.split_strategy

        if split_strategy == "train_test":
            model, scaler = load_model(models_dir / "dnn.pt", device=device)
            log.info("Loaded DNN model on %s", device)

        elif split_strategy == "k_fold":
            fold_paths = sorted(
                models_dir.glob("dnn_fold*.pt"),
                key=lambda p: int(re.search(r"\d+", p.stem).group()),
            )
            models_scalers = [load_model(p, device=device) for p in fold_paths]
            models = [ms[0] for ms in models_scalers]
            scalers = [ms[1] for ms in models_scalers]
            log.info("Loaded %d fold models", len(models))

        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

        # --- load predictions ---
        predictions_df = load_dataframe(dataframes_dir / "dnn_predictions.parquet")
        y_true = predictions_df["y_true"].to_numpy()
        y_pred = predictions_df["y_pred"].to_numpy()
        y_proba = predictions_df[[f"p_{name}" for name in class_names]].to_numpy()
        log.info("Predictions loaded: %d events", len(predictions_df))

        # --- feature importance (permutation) ---
        log.info("Computing permutation importance...")
        X, _, _ = prepare_features_target(df_mc)
        feature_names = X.columns.tolist()

        perm_model = model if split_strategy == "train_test" else models[0]
        perm_scaler = scaler if split_strategy == "train_test" else scalers[0]

        fig = plot_permutation_importance(
            perm_model,
            X,
            y_true,
            perm_scaler,
            device,
            feature_names=feature_names,
            n_features=20,
            seed=cfg.seed,
        )
        fi_path = plots_dir / "permutation_importance.png"
        save_figure(fig, fi_path)
        mlflow.log_artifact(str(fi_path))

        # --- shap ---
        log.info("Computing SHAP values (n_samples=2000)...")
        shap_model = model if split_strategy == "train_test" else models[0]
        shap_scaler = scaler if split_strategy == "train_test" else scalers[0]

        shap_values, X_sample = compute_dnn_shap_values(
            shap_model,
            X,
            shap_scaler,
            device,
            n_samples=2000,
            seed=cfg.seed,
        )
        log.info("SHAP computed on %d events", len(X_sample))

        fig = plot_shap_importance(
            shap_values, X_sample, class_labels=class_labels, n_features=20
        )
        shap_path = plots_dir / "shap_importance.png"
        save_figure(fig, shap_path)
        mlflow.log_artifact(str(shap_path))

        # --- summary metrics ---
        log.info("Computing summary metrics...")
        metrics = compute_summary_metrics(y_true, y_pred, y_proba, class_names)
        mlflow.log_metrics(metrics)
        for key, val in metrics.items():
            log.info("  %s: %.4f", key, val)

        metrics_path = metrics_dir / "dnn_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))

        # --- classification report ---
        log.info("Generating classification report...")
        fig = plot_classification_report(y_true, y_pred, class_labels=class_labels)
        cr_path = plots_dir / "classification_report.png"
        save_figure(fig, cr_path)
        mlflow.log_artifact(str(cr_path))

        # --- confusion matrix ---
        log.info("Generating confusion matrix...")
        fig = plot_confusion_matrix(y_true, y_pred, class_labels=class_labels)
        cm_path = plots_dir / "confusion_matrix.png"
        save_figure(fig, cm_path)
        mlflow.log_artifact(str(cm_path))

        # --- roc curves ---
        log.info("Generating ROC curves...")
        fig = plot_roc_curves(y_true, y_proba, class_labels=class_labels)
        roc_path = plots_dir / "roc_curves.png"
        save_figure(fig, roc_path)
        mlflow.log_artifact(str(roc_path))

        # --- pr curves ---
        log.info("Generating PR curves...")
        fig = plot_pr_curves(y_true, y_proba, class_labels=class_labels)
        pr_path = plots_dir / "pr_curves.png"
        save_figure(fig, pr_path)
        mlflow.log_artifact(str(pr_path))

        # --- score distributions ---
        log.info("Generating score distributions...")
        fig = plot_score_distributions(
            y_true, y_proba, class_labels=class_labels, bins=50
        )
        sd_path = plots_dir / "score_distributions.png"
        save_figure(fig, sd_path)
        mlflow.log_artifact(str(sd_path))

        log.info("Evaluation complete — plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
