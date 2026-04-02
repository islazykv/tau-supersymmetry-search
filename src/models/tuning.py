"""Optuna-based hyperparameter tuning for BDT and DNN models."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import xgboost as xgb
import yaml
from omegaconf import DictConfig, OmegaConf
from optuna.integration import XGBoostPruningCallback
from sklearn.preprocessing import MinMaxScaler

from src.models.bdt import build_params
from src.models.dnn import DNNClassifier, _batch_indices

# AMP: use modern API (PyTorch 2.3+) with fallback to legacy API
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.3
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.3

log = logging.getLogger(__name__)


class TqdmTrialCallback:
    """Optuna callback that displays a tqdm progress bar across trials."""

    def __init__(self, n_trials: int) -> None:
        self._bar = tqdm(total=n_trials, desc="Trial 0", unit="trial")

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self._bar.set_description(f"Trial {trial.number}")
        self._bar.update(1)

    def close(self) -> None:
        self._bar.close()


def suggest_params(
    trial: optuna.Trial,
    search_space: dict,
    model_name: str,
) -> dict:
    """Suggest parameters from a declarative YAML search-space config.

    Search-space types: float, int, categorical.
    """
    params = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(
                f"Unknown search space type {ptype!r} for {model_name}.{name}"
            )
    return params


def bdt_objective(
    trial: optuna.Trial,
    cfg: DictConfig,
    folds: list[tuple],
    n_classes: int,
) -> float:
    """XGBoost objective with per-round and cross-fold pruning."""
    search_space = OmegaConf.to_container(cfg.tuning.search_space.xgboost, resolve=True)
    suggested = suggest_params(trial, search_space, "xgboost")

    base_params = build_params(cfg, n_classes=n_classes)
    base_params.update(suggested)
    metric = base_params["eval_metric"]
    early_stopping_rounds = cfg.tuning.early_stopping_rounds

    fold_scores: list[float] = []

    fold_iter = (
        tqdm(folds, desc=f"Fold {trial.number}", leave=False)
        if len(folds) > 1
        else folds
    )

    for fold_idx, (X_tr, X_te, y_tr, y_te, w_tr, _) in enumerate(fold_iter):
        pruning_cb = XGBoostPruningCallback(trial, f"validation_1-{metric}")

        model = xgb.XGBClassifier(
            **base_params,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[pruning_cb],
        )

        fit_kwargs: dict = {
            "eval_set": [(X_tr, y_tr), (X_te, y_te)],
            "verbose": False,
        }
        if w_tr is not None:
            fit_kwargs["sample_weight"] = w_tr.to_numpy()

        model.fit(X_tr, y_tr, **fit_kwargs)
        fold_scores.append(model.best_score)

        # Cross-fold pruning: report running mean after each fold
        running_mean = np.mean(fold_scores)
        trial.report(running_mean, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned after fold {fold_idx + 1} (running mean: {running_mean:.6f})"
            )

    return float(np.mean(fold_scores))


def dnn_objective(
    trial: optuna.Trial,
    cfg: DictConfig,
    folds: list[tuple],
    n_classes: int,
    device: torch.device,
) -> float:
    """DNN objective with epoch-level pruning."""
    search_space = OmegaConf.to_container(cfg.tuning.search_space.dnn, resolve=True)
    suggested = suggest_params(trial, search_space, "dnn")

    n_layers = suggested.pop("n_layers")
    layer_size = suggested.pop("layer_size")
    hidden_sizes = [layer_size] * n_layers

    early_stopping_rounds = cfg.tuning.early_stopping_rounds
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    n_epochs: int = mcfg["n_epochs"]
    use_amp: bool = mcfg.get("amp", False) and device.type == "cuda"

    fold_best_losses: list[float] = []

    fold_iter = (
        tqdm(folds, desc=f"Fold {trial.number}", leave=False)
        if len(folds) > 1
        else folds
    )

    for fold_idx, (X_tr, X_te, y_tr, y_te, w_tr, _) in enumerate(fold_iter):
        n_features = X_tr.shape[1]

        best_val_loss = _train_dnn_with_pruning(
            trial=trial,
            fold_idx=fold_idx,
            n_folds=len(folds),
            X_tr=X_tr,
            X_te=X_te,
            y_tr=y_tr,
            y_te=y_te,
            w_tr=w_tr,
            n_features=n_features,
            n_classes=n_classes,
            hidden_sizes=hidden_sizes,
            dropout=suggested["dropout"],
            learning_rate=suggested["learning_rate"],
            weight_decay=suggested["weight_decay"],
            batch_size=suggested["batch_size"],
            n_epochs=n_epochs,
            early_stopping_rounds=early_stopping_rounds,
            use_amp=use_amp,
            device=device,
            seed=cfg.seed,
        )
        fold_best_losses.append(best_val_loss)

    return float(np.mean(fold_best_losses))


def _train_dnn_with_pruning(
    *,
    trial: optuna.Trial,
    fold_idx: int,
    n_folds: int,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    w_tr: pd.Series,
    n_features: int,
    n_classes: int,
    hidden_sizes: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    n_epochs: int,
    early_stopping_rounds: int,
    use_amp: bool,
    device: torch.device,
    seed: int,
) -> float:
    """Train a DNN for one fold with Optuna epoch-level pruning."""
    model = DNNClassifier(
        n_features=n_features,
        n_classes=n_classes,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    scaler = MinMaxScaler()
    scaler.fit(X_tr.values)

    class_weights = w_tr.groupby(y_tr).first().sort_index().values
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)

    X_train_t = torch.tensor(
        scaler.transform(X_tr.values), dtype=torch.float32, device=device
    )
    y_train_t = torch.tensor(y_tr.to_numpy(), dtype=torch.long, device=device)
    X_val_t = torch.tensor(
        scaler.transform(X_te.values), dtype=torch.float32, device=device
    )
    y_val_t = torch.tensor(y_te.to_numpy(), dtype=torch.long, device=device)
    n_train = len(X_train_t)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    grad_scaler = GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for idx in _batch_indices(n_train, batch_size, shuffle=True, generator=gen):
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(X_train_t[idx])
                loss = criterion(logits, y_train_t[idx])
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            running_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad(), autocast(enabled=use_amp):
            val_logits = model(X_val_t)
            epoch_val_loss = criterion(val_logits, y_val_t).item()

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Optuna pruning (epoch-level, globally indexed across folds) ---
        global_step = fold_idx * n_epochs + epoch
        trial.report(epoch_val_loss, global_step)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at fold {fold_idx + 1}/{n_folds}, epoch {epoch}"
            )

        if patience_counter >= early_stopping_rounds:
            break

    return best_val_loss


def create_study(cfg: DictConfig, storage_path: Path) -> optuna.Study:
    """Create or load an Optuna study with SQLite persistence.

    Samplers: TPESampler, RandomSampler.
    Pruners: MedianPruner, NopPruner.
    """
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    study_name = cfg.tuning.study_name
    direction = cfg.tuning.direction

    sampler_name = cfg.tuning.sampler
    if sampler_name == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    elif sampler_name == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name!r}")

    pruner_name = cfg.tuning.pruner
    if pruner_name == "MedianPruner":
        pruner = optuna.pruners.MedianPruner()
    elif pruner_name == "NopPruner":
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner_name!r}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )

    n_completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    if n_completed > 0:
        log.info("Resuming study with %d completed trials", n_completed)

    return study


def export_best_params(
    study: optuna.Study,
    model_name: str,
    base_model_cfg: DictConfig,
    output_path: Path,
) -> Path:
    """Export the best trial's params as a Hydra-compatible YAML model config."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged = OmegaConf.to_container(base_model_cfg, resolve=True)
    best_params = study.best_trial.params

    if model_name == "pytorch_dnn":
        n_layers = best_params.pop("n_layers", None)
        layer_size = best_params.pop("layer_size", None)
        if n_layers is not None and layer_size is not None:
            merged["hidden_sizes"] = [layer_size] * n_layers

    merged.update(best_params)

    with open(output_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    log.info("Best params exported to %s", output_path)
    return output_path
