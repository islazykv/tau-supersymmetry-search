"""Tests for model evaluation metrics.

Note: compute_summary_metrics uses label_binarize + multi_class="ovr" which
requires 3+ classes. The SUSY analysis always has 3+ classes (background types
+ signal), so binary tests are not included.
"""

from __future__ import annotations

import numpy as np

from src.models.evaluation import compute_summary_metrics


def _perfect_predictions(n: int, n_classes: int):
    """Return perfect (y_true, y_pred, y_proba) for n_classes."""
    y_true = np.tile(np.arange(n_classes), n // n_classes + 1)[:n]
    y_pred = y_true.copy()
    y_proba = np.zeros((n, n_classes))
    for i in range(n):
        y_proba[i, y_true[i]] = 1.0
    return y_true, y_pred, y_proba


class TestComputeSummaryMetrics:
    def test_multiclass_keys(self) -> None:
        rng = np.random.RandomState(0)
        n, nc = 150, 3
        y_true = np.tile([0, 1, 2], n // 3)
        y_pred = rng.randint(0, nc, size=n)
        y_proba = rng.dirichlet([1] * nc, size=n)
        class_names = ["ttbar", "Ztt", "signal"]

        metrics = compute_summary_metrics(y_true, y_pred, y_proba, class_names)
        expected_keys = {
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "roc_auc_macro",
            "auc_ttbar",
            "auc_Ztt",
            "auc_signal",
        }
        assert expected_keys == set(metrics.keys())

    def test_metric_ranges(self) -> None:
        rng = np.random.RandomState(42)
        n, nc = 300, 4
        y_true = np.tile(np.arange(nc), n // nc)
        y_pred = rng.randint(0, nc, size=n)
        y_proba = rng.dirichlet([1] * nc, size=n)
        class_names = ["a", "b", "c", "d"]

        metrics = compute_summary_metrics(y_true, y_pred, y_proba, class_names)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_perfect_predictions(self) -> None:
        y_true, y_pred, y_proba = _perfect_predictions(150, 3)
        metrics = compute_summary_metrics(y_true, y_pred, y_proba, ["a", "b", "c"])
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc_macro"] == 1.0

    def test_per_class_auc_count(self) -> None:
        rng = np.random.RandomState(7)
        n, nc = 200, 5
        y_true = np.tile(np.arange(nc), n // nc)
        y_pred = rng.randint(0, nc, size=n)
        y_proba = rng.dirichlet([1] * nc, size=n)
        class_names = [f"cls_{i}" for i in range(nc)]

        metrics = compute_summary_metrics(y_true, y_pred, y_proba, class_names)
        auc_keys = [k for k in metrics if k.startswith("auc_")]
        assert len(auc_keys) == nc
