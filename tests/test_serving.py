"""Tests for the serving API using real tiny in-memory models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.registry import BDTAdapter, DNNAdapter

FEATURES = ["feat_a", "feat_b", "feat_c"]
CLASS_NAMES = ["background", "signal"]


@pytest.fixture()
def bdt_path(tmp_path: Path) -> Path:
    """Train a tiny 2-class XGBoost model and save it."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(60, 3), columns=FEATURES)
    y = pd.Series(rng.randint(0, 2, size=60))
    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=2,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X, y)
    path = tmp_path / "tiny.ubj"
    model.save_model(str(path))
    return path


@pytest.fixture()
def dnn_path(tmp_path: Path) -> Path:
    """Train a tiny 2-class DNN and save checkpoint with scaler."""
    import torch
    from sklearn.preprocessing import MinMaxScaler

    from src.models.dnn import DNNClassifier

    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(60, 3), columns=FEATURES)
    y = pd.Series(rng.randint(0, 2, size=60))

    scaler = MinMaxScaler()
    scaler.fit(X)  # DataFrame → stores feature_names_in_

    model = DNNClassifier(n_features=3, n_classes=2, hidden_sizes=[8])
    # Quick 1-epoch train so weights are not random
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    X_t = torch.tensor(scaler.transform(X.values), dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.long)
    model.train()
    logits = model(X_t)
    loss = criterion(logits, y_t)
    loss.backward()
    optimizer.step()
    model.eval()

    path = tmp_path / "tiny.pt"
    torch.save(
        {
            "model_config": model.config,
            "state_dict": model.state_dict(),
        },
        str(path),
    )
    import joblib

    joblib.dump(scaler, str(path.with_suffix(".scaler")))
    return path


class TestBDTAdapter:
    def test_feature_names(self, bdt_path: Path) -> None:
        adapter = BDTAdapter(bdt_path)
        assert adapter.feature_names() == FEATURES

    def test_n_classes(self, bdt_path: Path) -> None:
        adapter = BDTAdapter(bdt_path)
        assert adapter.n_classes() == 2

    def test_class_names_default(self, bdt_path: Path) -> None:
        adapter = BDTAdapter(bdt_path)
        assert adapter.class_names() == ["0", "1"]

    def test_class_names_custom(self, bdt_path: Path) -> None:
        adapter = BDTAdapter(bdt_path, class_names=CLASS_NAMES)
        assert adapter.class_names() == CLASS_NAMES

    def test_predict_proba_shape(self, bdt_path: Path) -> None:
        adapter = BDTAdapter(bdt_path)
        X = pd.DataFrame(np.zeros((3, 3)), columns=FEATURES)
        proba = adapter.predict_proba(X)
        assert proba.shape == (3, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestDNNAdapter:
    def test_feature_names(self, dnn_path: Path) -> None:
        adapter = DNNAdapter(dnn_path)
        assert adapter.feature_names() == FEATURES

    def test_n_classes(self, dnn_path: Path) -> None:
        adapter = DNNAdapter(dnn_path)
        assert adapter.n_classes() == 2

    def test_predict_proba_shape(self, dnn_path: Path) -> None:
        adapter = DNNAdapter(dnn_path)
        X = pd.DataFrame(np.zeros((3, 3)), columns=FEATURES)
        proba = adapter.predict_proba(X)
        assert proba.shape == (3, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.fixture()
def bdt_client(bdt_path: Path) -> TestClient:
    """TestClient with a BDT model loaded."""
    app = create_app(
        model_type="bdt",
        model_path=str(bdt_path),
        class_names=CLASS_NAMES,
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def dnn_client(dnn_path: Path) -> TestClient:
    """TestClient with a DNN model loaded."""
    app = create_app(
        model_type="dnn",
        model_path=str(dnn_path),
        class_names=CLASS_NAMES,
    )
    with TestClient(app) as client:
        yield client


class TestHealth:
    def test_health_ok(self, bdt_client: TestClient) -> None:
        resp = bdt_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestModelInfo:
    def test_info_bdt(self, bdt_client: TestClient) -> None:
        resp = bdt_client.get("/v1/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "bdt"
        assert data["features"] == FEATURES
        assert data["n_classes"] == 2
        assert data["class_names"] == CLASS_NAMES

    def test_info_dnn(self, dnn_client: TestClient) -> None:
        resp = dnn_client.get("/v1/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "dnn"
        assert data["features"] == FEATURES


class TestPredict:
    def test_single_prediction(self, bdt_client: TestClient) -> None:
        resp = bdt_client.post(
            "/v1/predict",
            json={"features": {"feat_a": 0.5, "feat_b": -1.0, "feat_c": 0.0}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_class" in data
        assert data["predicted_label"] in CLASS_NAMES
        assert set(data["probabilities"].keys()) == set(CLASS_NAMES)
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 1e-5

    def test_missing_feature(self, bdt_client: TestClient) -> None:
        resp = bdt_client.post(
            "/v1/predict",
            json={"features": {"feat_a": 0.5}},
        )
        assert resp.status_code == 422
        assert "Missing features" in resp.json()["detail"]

    def test_extra_features_ignored(self, bdt_client: TestClient) -> None:
        resp = bdt_client.post(
            "/v1/predict",
            json={
                "features": {
                    "feat_a": 0.5,
                    "feat_b": -1.0,
                    "feat_c": 0.0,
                    "extra": 999.0,
                }
            },
        )
        assert resp.status_code == 200

    def test_dnn_prediction(self, dnn_client: TestClient) -> None:
        resp = dnn_client.post(
            "/v1/predict",
            json={"features": {"feat_a": 0.1, "feat_b": 0.2, "feat_c": 0.3}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_label"] in CLASS_NAMES


class TestBatchPredict:
    def test_batch(self, bdt_client: TestClient) -> None:
        samples = [
            {"feat_a": 0.5, "feat_b": -1.0, "feat_c": 0.0},
            {"feat_a": -0.3, "feat_b": 2.0, "feat_c": 1.0},
        ]
        resp = bdt_client.post("/v1/predict/batch", json={"samples": samples})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert pred["predicted_label"] in CLASS_NAMES

    def test_batch_missing_feature(self, bdt_client: TestClient) -> None:
        samples = [
            {"feat_a": 0.5, "feat_b": -1.0, "feat_c": 0.0},
            {"feat_a": 0.5},  # missing feat_b, feat_c
        ]
        resp = bdt_client.post("/v1/predict/batch", json={"samples": samples})
        assert resp.status_code == 422
