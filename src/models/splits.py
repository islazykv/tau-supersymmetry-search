from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as _sklearn_split

from src.processing.validation import METADATA_COLUMNS


def prepare_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate training features, class labels, and sample weights from the MC DataFrame."""
    training_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    X = df[training_cols].copy()
    y = df["class"].copy()
    weights = df["class_weight"].copy()
    return X, y, weights


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    test_size: float = 0.2,
    seed: int = 1,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Stratified train/test split that keeps features, labels, and weights aligned."""
    X_train, X_test, y_train, y_test, w_train, w_test = _sklearn_split(
        X,
        y,
        weights,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, w_train, w_test


def kfold_split(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    n_splits: int = 5,
    seed: int = 1,
) -> list[
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]
]:
    """Stratified K-fold split returning one (X_train, X_test, y_train, y_test, w_train, w_test) tuple per fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(X, y):
        folds.append(
            (
                X.iloc[train_idx],
                X.iloc[test_idx],
                y.iloc[train_idx],
                y.iloc[test_idx],
                weights.iloc[train_idx],
                weights.iloc[test_idx],
            )
        )
    return folds


def build_predictions_frame(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Assemble a tidy DataFrame of true labels, hard predictions, and per-class probabilities."""
    df = pd.DataFrame(
        {
            "y_true": y_true.to_numpy(),
            "y_pred": y_pred,
        }
    )
    for i, name in enumerate(class_names):
        df[f"p_{name}"] = y_proba[:, i]
    return df
