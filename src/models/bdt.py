from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.models.splits import build_predictions_frame  # noqa: F401

log = logging.getLogger(__name__)
_MULTICLASS_OBJECTIVE = "multi:softproba"
_MULTICLASS_METRIC = "mlogloss"
_BINARY_OBJECTIVE = "binary:logistic"
_BINARY_METRIC = "logloss"


class _TqdmCallback(xgb.callback.TrainingCallback):
    """Updates a pre-created tqdm bar after each boosting round."""

    def __init__(self, bar: tqdm) -> None:
        super().__init__()
        self._bar = bar

    def after_iteration(self, model, epoch, evals_log):
        postfix = {}
        for name, metrics in evals_log.items():
            tag = "train" if name == "validation_0" else "val"
            for metric, values in metrics.items():
                postfix[f"{tag}_{metric}"] = f"{values[-1]:.4f}"
        self._bar.set_postfix(postfix)
        self._bar.update(1)
        return False

    def after_training(self, model):
        return model


def build_params(cfg: DictConfig, n_classes: int) -> dict:
    """Build XGBoost constructor kwargs from the Hydra model config.

    Reads all keys from ``cfg.model`` (excluding ``name``), then injects the
    correct objective and eval metric based on the number of classes.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config object.
    n_classes : int
        Number of unique class labels in the dataset.

    Returns
    -------
    dict
        Keyword arguments ready to be unpacked into ``xgb.XGBClassifier``.
    """
    params: dict = {
        k: v
        for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
        if k != "name"
    }

    if n_classes > 2:
        params["objective"] = _MULTICLASS_OBJECTIVE
        params["eval_metric"] = _MULTICLASS_METRIC
        params["num_class"] = n_classes
    else:
        params["objective"] = _BINARY_OBJECTIVE
        params["eval_metric"] = _BINARY_METRIC

    return params


def train(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_train: pd.Series | None = None,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with early stopping on the validation set.

    Both the training set and the validation set are passed as eval sets so
    that ``evals_result_`` contains curves for both.  XGBoost monitors only
    the *last* entry in ``eval_set`` for early stopping, which is the
    validation set.

    Parameters
    ----------
    params : dict
        XGBoost constructor kwargs produced by :func:`build_params`.
    X_train, y_train : pd.DataFrame / pd.Series
        Training features and labels.
    X_val, y_val : pd.DataFrame / pd.Series
        Validation features and labels (used for early stopping).
    w_train : pd.Series, optional
        Per-event sample weights for class imbalance correction.
    early_stopping_rounds : int
        Stop training if validation metric does not improve for this many
        consecutive rounds.

    Returns
    -------
    xgb.XGBClassifier
        Fitted model.  ``model.best_iteration`` holds the optimal number of
        trees; predictions automatically use this limit.
    """
    model = xgb.XGBClassifier(
        **params,
        early_stopping_rounds=early_stopping_rounds,
    )

    fit_kwargs: dict = {
        "eval_set": [(X_train, y_train), (X_val, y_val)],
        "verbose": False,
    }
    if w_train is not None:
        fit_kwargs["sample_weight"] = w_train.to_numpy()

    pbar = (
        tqdm(total=params.get("n_estimators", 100), desc="Training", unit="tree")
        if verbose
        else None
    )
    if pbar is not None:
        fit_kwargs["callbacks"] = [_TqdmCallback(pbar)]

    model.fit(X_train, y_train, **fit_kwargs)

    if pbar is not None:
        pbar.close()
    log.info(
        "Training complete — best iteration: %d, best score: %.6f",
        model.best_iteration,
        model.best_score,
    )
    return model


def train_kfold(
    params: dict,
    folds: list[
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]
    ],
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> tuple[list[xgb.XGBClassifier], np.ndarray, np.ndarray, pd.Series]:
    """Train one model per fold and return out-of-fold (OOF) predictions.

    Each fold's test set is non-overlapping, so concatenating OOF predictions
    gives full-dataset coverage — a robust, unbiased estimate of generalisation.

    Parameters
    ----------
    params : dict
        XGBoost constructor kwargs produced by :func:`build_params`.
    folds : list of tuples
        Output of :func:`~src.models.splits.kfold_split`.
    early_stopping_rounds : int
        Passed to each fold's :func:`train` call.

    Returns
    -------
    models : list[xgb.XGBClassifier]
        One fitted model per fold.
    y_pred_oof : np.ndarray, shape (n_samples,)
        Concatenated hard predictions over all folds.
    y_proba_oof : np.ndarray, shape (n_samples, n_classes)
        Concatenated probability scores over all folds.
    y_test_oof : pd.Series
        Concatenated true labels over all folds (same row order as predictions).
    """
    models: list[xgb.XGBClassifier] = []
    y_preds: list[np.ndarray] = []
    y_probas: list[np.ndarray] = []
    y_tests: list[pd.Series] = []

    fold_pbar = tqdm(total=len(folds), desc="Folds", unit="fold") if verbose else None
    for fold_idx, (X_tr, X_te, y_tr, y_te, w_tr, _) in enumerate(folds):
        log.info("Training fold %d / %d", fold_idx + 1, len(folds))
        model = train(
            params.copy(),
            X_tr,
            y_tr,
            X_te,
            y_te,
            w_train=w_tr,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        y_pred_fold, y_proba_fold = predict(model, X_te)

        models.append(model)
        y_preds.append(y_pred_fold)
        y_probas.append(y_proba_fold)
        y_tests.append(y_te)
        if fold_pbar is not None:
            fold_pbar.update(1)

    if fold_pbar is not None:
        fold_pbar.close()

    y_pred_oof = np.concatenate(y_preds)
    y_proba_oof = np.vstack(y_probas)
    y_test_oof = pd.concat(y_tests)

    return models, y_pred_oof, y_proba_oof, y_test_oof


def get_evals_result(model: xgb.XGBClassifier) -> dict[str, dict[str, list[float]]]:
    """Return the per-iteration evaluation results stored on the fitted model.

    The returned dict has the shape::

        {
            "validation_0": {"<metric>": [v0, v1, ...]},  # train set
            "validation_1": {"<metric>": [v0, v1, ...]},  # val set
        }
    """
    return model.evals_result()


def predict(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hard predictions and probability scores for *X*.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Fitted model.
    X : pd.DataFrame
        Input features.

    Returns
    -------
    y_pred : np.ndarray, shape (n_samples,)
        Argmax class indices.
    y_proba : np.ndarray, shape (n_samples, n_classes)
        Per-class probability estimates.
    """
    y_proba: np.ndarray = model.predict_proba(X)
    y_pred: np.ndarray = np.argmax(y_proba, axis=1)
    return y_pred, y_proba


def save_model(model: xgb.XGBClassifier, path: Path) -> None:
    """Save the XGBoost model in its native binary format (.ubj).

    The native format preserves all model metadata including the best
    iteration, so the loaded model behaves identically to the original.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    log.info("Model saved to %s", path)


def load_model(path: Path) -> xgb.XGBClassifier:
    """Load an XGBoost model from a native binary file."""
    model = xgb.XGBClassifier()
    model.load_model(str(Path(path)))
    log.info("Model loaded from %s", path)
    return model
