"""Smoke tests for BDT and DNN training, prediction, and serialization."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import xgboost as xgb
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler

from src.models.bdt import (
    build_params,
    get_evals_result,
    load_model as bdt_load_model,
    predict as bdt_predict,
    save_model as bdt_save_model,
    train as bdt_train,
)
from src.models.dnn import (
    DNNClassifier,
    build_criterion,
    build_scaler,
    load_model as dnn_load_model,
    predict as dnn_predict,
    save_model as dnn_save_model,
    train as dnn_train,
)

FEATURES = ["f0", "f1", "f2", "f3", "f4"]
N_SAMPLES = 60


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic 2-class train/val split."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(N_SAMPLES, len(FEATURES)), columns=FEATURES)
    y = pd.Series(rng.randint(0, 2, size=N_SAMPLES))
    split = N_SAMPLES // 2
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


class TestBDTBuildParams:
    def test_binary(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"name": "xgboost", "n_estimators": 5, "max_depth": 2}}
        )
        params = build_params(cfg, n_classes=2)
        assert params["objective"] == "binary:logistic"
        assert params["eval_metric"] == "logloss"
        assert "num_class" not in params

    def test_multiclass(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"name": "xgboost", "n_estimators": 5, "max_depth": 2}}
        )
        params = build_params(cfg, n_classes=4)
        assert params["objective"] == "multi:softproba"
        assert params["num_class"] == 4


class TestBDTTrainPredict:
    def test_train_and_predict(self, synthetic_data) -> None:
        X_tr, X_val, y_tr, y_val = synthetic_data
        params = {
            "n_estimators": 5,
            "max_depth": 2,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        model = bdt_train(params, X_tr, y_tr, X_val, y_val, verbose=False)

        assert isinstance(model, xgb.XGBClassifier)

        y_pred, y_proba = bdt_predict(model, X_val)
        assert y_pred.shape == (len(X_val),)
        assert y_proba.shape == (len(X_val), 2)
        np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)

    def test_evals_result(self, synthetic_data) -> None:
        X_tr, X_val, y_tr, y_val = synthetic_data
        params = {
            "n_estimators": 5,
            "max_depth": 2,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        model = bdt_train(params, X_tr, y_tr, X_val, y_val, verbose=False)

        evals = get_evals_result(model)
        assert "validation_0" in evals
        assert "validation_1" in evals


class TestBDTSerialization:
    def test_save_load_roundtrip(self, synthetic_data, tmp_path: Path) -> None:
        X_tr, X_val, y_tr, y_val = synthetic_data
        params = {
            "n_estimators": 5,
            "max_depth": 2,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        model = bdt_train(params, X_tr, y_tr, X_val, y_val, verbose=False)
        _, proba_before = bdt_predict(model, X_val)

        path = tmp_path / "model.ubj"
        bdt_save_model(model, path)
        loaded = bdt_load_model(path)
        _, proba_after = bdt_predict(loaded, X_val)

        np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)


class TestDNNClassifier:
    def test_forward_shape(self) -> None:
        model = DNNClassifier(n_features=5, n_classes=3, hidden_sizes=[8, 4])
        x = torch.randn(10, 5)
        out = model(x)
        assert out.shape == (10, 3)

    def test_config_preserved(self) -> None:
        model = DNNClassifier(n_features=5, n_classes=2, hidden_sizes=[16], dropout=0.1)
        assert model.config["n_features"] == 5
        assert model.config["n_classes"] == 2
        assert model.config["hidden_sizes"] == [16]


class TestDNNHelpers:
    def test_build_scaler(self, synthetic_data) -> None:
        X_tr, _, _, _ = synthetic_data
        scaler = build_scaler(X_tr)
        assert isinstance(scaler, MinMaxScaler)
        transformed = scaler.transform(X_tr)
        assert transformed.min() >= -1e-10
        assert transformed.max() <= 1.0 + 1e-10

    def test_build_criterion_no_weights(self) -> None:
        criterion = build_criterion()
        assert isinstance(criterion, torch.nn.CrossEntropyLoss)

    def test_build_criterion_with_weights(self) -> None:
        weights = np.array([1.0, 0.5, 0.3])
        criterion = build_criterion(class_weights=weights)
        assert criterion.weight is not None
        assert criterion.weight.shape == (3,)


class TestDNNTrainPredict:
    def test_train_and_predict(self, synthetic_data) -> None:
        X_tr, X_val, y_tr, y_val = synthetic_data
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "model": {
                    "name": "pytorch_dnn",
                    "hidden_sizes": [8],
                    "activation": "ReLU",
                    "dropout": 0.0,
                    "n_epochs": 3,
                    "batch_size": 32,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "amp": False,
                },
            }
        )
        model = DNNClassifier(n_features=len(FEATURES), n_classes=2, hidden_sizes=[8])
        scaler = build_scaler(X_tr)
        criterion = build_criterion()
        device = torch.device("cpu")

        trained_model, history = dnn_train(
            model,
            criterion,
            X_tr,
            y_tr,
            X_val,
            y_val,
            scaler,
            cfg,
            device,
            early_stopping_rounds=5,
            verbose=False,
        )
        assert "train_loss" in history
        assert "val_loss" in history

        y_pred, y_proba = dnn_predict(trained_model, X_val, scaler, device)
        assert y_pred.shape == (len(X_val),)
        assert y_proba.shape == (len(X_val), 2)
        np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)


class TestDNNSerialization:
    def test_save_load_roundtrip(self, synthetic_data, tmp_path: Path) -> None:
        X_tr, X_val, _, _ = synthetic_data
        model = DNNClassifier(n_features=len(FEATURES), n_classes=2, hidden_sizes=[8])
        scaler = build_scaler(X_tr)
        device = torch.device("cpu")

        _, proba_before = dnn_predict(model, X_val, scaler, device)

        path = tmp_path / "model.pt"
        dnn_save_model(model, scaler, path)

        assert path.exists()
        assert path.with_suffix(".scaler").exists()

        loaded_model, loaded_scaler = dnn_load_model(path)
        _, proba_after = dnn_predict(loaded_model, X_val, loaded_scaler, device)

        np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)

    def test_weights_only_true(self, tmp_path: Path) -> None:
        """Verify the checkpoint loads with weights_only=True (no pickle)."""
        model = DNNClassifier(n_features=3, n_classes=2, hidden_sizes=[4])
        scaler = MinMaxScaler()
        scaler.fit(np.random.randn(10, 3))

        path = tmp_path / "safe.pt"
        dnn_save_model(model, scaler, path)

        # Should not raise — weights_only=True is safe because scaler is separate
        checkpoint = torch.load(str(path), weights_only=True)
        assert "state_dict" in checkpoint
        assert "model_config" in checkpoint
        assert "scaler" not in checkpoint  # scaler is in .scaler file

        loaded_scaler = joblib.load(path.with_suffix(".scaler"))
        assert isinstance(loaded_scaler, MinMaxScaler)
