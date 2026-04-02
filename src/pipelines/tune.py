"""Hyperparameter tuning pipeline with Optuna and MLflow."""

from __future__ import annotations

import logging

import mlflow
import optuna
import pyrootutils
from omegaconf import DictConfig, OmegaConf

from src.eda.utils import get_class_names
from src.models.dnn import resolve_device
from src.models.splits import kfold_split, prepare_features_target, train_test_split
from src.models.tuning import (
    TqdmTrialCallback,
    bdt_objective,
    create_study,
    dnn_objective,
    export_best_params,
)
from src.processing.analysis import get_output_paths
from src.processing.io import load_dataframe

log = logging.getLogger(__name__)


def tune(cfg: DictConfig) -> None:
    """Run Optuna hyperparameter tuning with k-fold CV and MLflow tracking."""
    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])

    log.info("Starting hyperparameter tuning:\n%s", OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(f"file://{root}/mlruns")
    mlflow.set_experiment(cfg.get("experiment_name", "tau-supersymmetry-search-tuning"))

    with mlflow.start_run(run_name=f"tuning-{cfg.model.name}"):
        mlflow.log_params(
            {
                "model_name": cfg.model.name,
                "n_trials": cfg.tuning.n_trials,
                "n_splits": cfg.tuning.n_splits,
                "sampler": cfg.tuning.sampler,
                "pruner": cfg.tuning.pruner,
                "early_stopping_rounds": cfg.tuning.early_stopping_rounds,
                "seed": cfg.seed,
            }
        )

        output_paths = get_output_paths(cfg)
        dataframes_dir = root / output_paths["dataframes_dir"]
        models_dir = root / output_paths["models_dir"]
        models_dir.mkdir(parents=True, exist_ok=True)

        df_mc = load_dataframe(dataframes_dir / "mc.parquet")
        log.info("Loaded MC: %d events, %d columns", len(df_mc), len(df_mc.columns))

        class_names = get_class_names(df_mc)
        n_classes = len(class_names)
        log.info("Classes (%d): %s", n_classes, class_names)

        subsample = cfg.tuning.get("subsample_fraction", 1.0)
        if subsample < 1.0:
            df_mc = df_mc.groupby("class", group_keys=False).apply(
                lambda g: g.sample(frac=subsample, random_state=cfg.seed)
            )
            log.info("Subsampled to %.0f%%: %d events", subsample * 100, len(df_mc))

        X, y, weights = prepare_features_target(df_mc)
        mlflow.log_param("n_features", X.shape[1])

        split_method = cfg.tuning.get("split_method", "kfold")
        if split_method == "kfold":
            n_splits = cfg.tuning.n_splits
            folds = kfold_split(X, y, weights, n_splits=n_splits, seed=cfg.seed)
            log.info("K-fold: %d stratified folds", n_splits)
        else:
            test_size = cfg.tuning.get("test_size", 0.2)
            folds = [
                train_test_split(X, y, weights, test_size=test_size, seed=cfg.seed)
            ]
            log.info("Train-test split: %.0f%% test", test_size * 100)

        storage_path = models_dir / cfg.tuning.storage_filename
        study = create_study(cfg, storage_path)

        try:
            from optuna_integration.mlflow import MLflowCallback

            mlflow_cb = MLflowCallback(
                tracking_uri=f"file://{root}/mlruns",
                metric_name="mean_cv_loss",
                create_experiment=False,
                mlflow_kwargs={"nested": True},
            )
            callbacks = [mlflow_cb]
        except ImportError:
            log.warning(
                "optuna-integration not available; skipping MLflow trial logging"
            )
            callbacks = []

        model_name = cfg.model.name
        n_trials = cfg.tuning.n_trials
        tqdm_cb = TqdmTrialCallback(n_trials)
        callbacks.append(tqdm_cb)

        if model_name == "xgboost":
            study.optimize(
                lambda trial: bdt_objective(trial, cfg, folds, n_classes),
                n_trials=n_trials,
                callbacks=callbacks,
            )
        elif model_name == "pytorch_dnn":
            device = resolve_device()
            log.info("Device: %s", device)
            study.optimize(
                lambda trial: dnn_objective(trial, cfg, folds, n_classes, device),
                n_trials=n_trials,
                callbacks=callbacks,
            )
        else:
            raise ValueError(f"Unknown model: {model_name!r}")

        log.info("Study complete: %d trials", len(study.trials))
        n_completed = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        n_pruned = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        )
        log.info("Completed: %d | Pruned: %d", n_completed, n_pruned)

        if n_completed > 0:
            log.info(
                "Best trial #%d — value: %.6f",
                study.best_trial.number,
                study.best_trial.value,
            )
            log.info("Best params: %s", study.best_trial.params)

            mlflow.log_metric("best_trial_value", study.best_trial.value)
            mlflow.log_params(
                {f"best_{k}": v for k, v in study.best_trial.params.items()}
            )

            suffix = "xgboost" if model_name == "xgboost" else "dnn"
            params_path = models_dir / f"{suffix}_best_params.yaml"
            export_best_params(study, model_name, cfg.model, params_path)
            mlflow.log_artifact(str(params_path))

        mlflow.log_artifact(str(storage_path))

        log.info("Tuning complete — results saved to %s", models_dir)
