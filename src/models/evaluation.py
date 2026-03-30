from __future__ import annotations

import atlas_mpl_style as ampl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

_LEGEND_KW: dict = dict(
    loc="upper right",
    prop={"size": 12},
    labelspacing=0.1,
    handlelength=1,
    handleheight=1,
)


def compute_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Accuracy, macro/weighted F1, macro OvR AUC, and per-class AUC as a flat dict."""
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "roc_auc_macro": float(
            roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        ),
    }
    for i, name in enumerate(class_names):
        metrics[f"auc_{name}"] = float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
    return metrics


def _latex_safe(name: str) -> str:
    """Escape LaTeX special characters in a plain-text string."""
    return name.replace("_", r"\_")


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    n_features: int = 20,
) -> plt.Figure:
    """Horizontal bar chart of the top-N XGBoost built-in feature importances."""
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-n_features:]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top_idx) * 0.45)))
    ax.barh(range(len(top_idx)), importances[top_idx])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([_latex_safe(feature_names[i]) for i in top_idx], fontsize=10)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def compute_shap_values(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Compute SHAP values on a random subsample of X; return (shap_values, X_sample)."""
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sample = X.iloc[idx].reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, X_sample


def plot_shap_importance(
    shap_values: list[np.ndarray],
    X_sample: pd.DataFrame,
    class_labels: list[str],
    n_features: int = 20,
) -> plt.Figure:
    """SHAP multiclass bar summary showing per-class mean |SHAP| for the top features."""
    import shap

    X_display = X_sample.rename(columns=_latex_safe)
    shap.summary_plot(
        shap_values,
        X_display,
        plot_type="bar",
        class_names=class_labels,
        max_display=n_features,
        show=False,
    )
    fig = plt.gcf()
    ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=fig.axes[0])
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> plt.Figure:
    """Row-normalised confusion matrix; cells show event count and row percentage."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array(
        [f"{c:,}\n{p:.1f}%" for c, p in zip(cm.flatten(), cm_norm.flatten())]
    ).reshape(cm.shape)

    figsize = 14 if len(class_labels) <= 8 else 20
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    sns.heatmap(
        cm_norm,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="Greens",
        linewidths=0.5,
        linecolor="black",
        square=True,
        vmin=0,
        vmax=100,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"format": "%.0f%%", "shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
) -> plt.Figure:
    """Heatmap of per-class precision, recall, and F1-score alongside support counts."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_labels))),
        target_names=class_labels,
        output_dict=True,
    )
    df = pd.DataFrame(report).T
    metrics_df = df.loc[class_labels, ["precision", "recall", "f1-score"]] * 100
    support_df = df.loc[class_labels, ["support"]]

    n = len(class_labels)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, max(4, n * 0.8 + 2)),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    sns.heatmap(
        metrics_df,
        ax=axes[0],
        annot=True,
        cmap="Greens",
        linewidth=0.5,
        linecolor="black",
        fmt=".1f",
        vmin=0,
        vmax=100,
        cbar_kws={"format": "%.0f%%"},
    )
    axes[0].set_title("Classification Report (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(
        support_df,
        ax=axes[1],
        annot=True,
        cmap="Blues",
        linewidth=0.5,
        linecolor="black",
        fmt=".0f",
        cbar=True,
    )
    axes[1].set_title("Support")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=axes[0])
    fig.tight_layout()
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: list[str],
) -> plt.Figure:
    """Per-class one-vs-rest ROC curves with AUC values in the legend."""
    y_bin = label_binarize(y_true, classes=range(len(class_labels)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        score = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{label} (AUC = {score:.3f})")
    ax.plot(
        [0, 1], [0, 1], "--", color="grey", linewidth=1.0, label="Random (AUC = 0.500)"
    )
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (one-vs-rest)")
    leg = ax.legend(
        fontsize=10, bbox_to_anchor=(1.01, 1.0), loc="upper left", frameon=True
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)
    ax.grid(True, alpha=0.3)
    ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: list[str],
) -> plt.Figure:
    """Per-class precision-recall curves with average precision in the legend."""
    y_bin = label_binarize(y_true, classes=range(len(class_labels)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, label in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax.plot(recall, precision, linewidth=1.5, label=f"{label} (AP = {ap:.3f})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (one-vs-rest)")
    leg = ax.legend(
        fontsize=10, bbox_to_anchor=(1.01, 1.0), loc="upper left", frameon=True
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)
    ax.grid(True, alpha=0.3)
    ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def plot_score_distributions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: list[str],
    bins: int = 50,
) -> plt.Figure:
    """For each output node, normalised score distributions split by true class."""
    n = len(class_labels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for i, (label, ax) in enumerate(zip(class_labels, axes_flat)):
        for j in range(n):
            mask = y_true == j
            ax.hist(
                y_proba[mask, i],
                bins=bins,
                density=True,
                alpha=0.6,
                histtype="step",
                linewidth=1.2,
                label=class_labels[j],
            )
        ax.set_xlabel(f"P({label})")
        ax.set_ylabel("Normalised Events")
        ax.set_title(f"Score Distribution — {label}")
        ax.legend(prop={"size": 8}, labelspacing=0.1, handlelength=1, handleheight=1)
        ax.grid(True, alpha=0.3)
        ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=ax)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    scaler,
    device,
    feature_names: list[str],
    n_features: int = 20,
    n_samples: int = 5000,
    n_repeats: int = 5,
    seed: int = 42,
) -> plt.Figure:
    """Permutation importance for a DNN model, plotted as a horizontal bar chart.

    Uses a subsample of ``X`` for speed.  The scoring function is accuracy.
    """
    from sklearn.inspection import permutation_importance as _perm_importance

    from src.models.dnn import predict as dnn_predict

    rng = np.random.default_rng(seed)
    n_available = min(len(X), len(y))
    idx = rng.choice(n_available, size=min(n_samples, n_available), replace=False)
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]

    class _ScoringWrapper:
        """Wrap DNN predict into a sklearn-compatible estimator for permutation_importance."""

        def __init__(self, model, scaler, device):
            self._model = model
            self._scaler = scaler
            self._device = device
            self.classes_ = np.arange(model.config["n_classes"])

        def fit(self, X, y):
            return self

        def predict(self, X):
            y_pred, _ = dnn_predict(
                self._model,
                pd.DataFrame(X, columns=feature_names),
                self._scaler,
                self._device,
            )
            return y_pred

    wrapper = _ScoringWrapper(model, scaler, device)

    result = _perm_importance(
        wrapper,
        X_sub,
        y_sub,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="accuracy",
    )

    importances = result.importances_mean
    top_idx = np.argsort(importances)[-n_features:]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top_idx) * 0.45)))
    ax.barh(range(len(top_idx)), importances[top_idx])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([_latex_safe(feature_names[i]) for i in top_idx], fontsize=10)
    ax.set_xlabel("Mean Accuracy Decrease")
    ax.set_title("Permutation Importance")
    ax.grid(True, alpha=0.3, axis="x")
    ampl.draw_atlas_label(0.02, 0.97, simulation=True, status="final", ax=ax)
    fig.tight_layout()
    return fig


def compute_dnn_shap_values(
    model,
    X: pd.DataFrame,
    scaler,
    device,
    n_samples: int = 2000,
    n_background: int = 100,
    seed: int = 42,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Compute SHAP values for a DNN using GradientExplainer.

    Parameters
    ----------
    model : DNNClassifier
        Trained PyTorch model.
    X : pd.DataFrame
        Full feature DataFrame (pre-scaling).
    scaler : MinMaxScaler
        Fitted scaler.
    device : torch.device
        Model device.
    n_samples : int
        Number of events to explain.
    n_background : int
        Number of background samples for the explainer.
    seed : int
        Random seed.

    Returns
    -------
    shap_values : list[np.ndarray]
        Per-class SHAP values, each of shape ``(n_samples, n_features)``.
    X_sample : pd.DataFrame
        The (unscaled) feature subsample used for explanations.
    """
    import shap
    import torch

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    bg_idx = rng.choice(len(X), size=min(n_background, len(X)), replace=False)

    X_sample = X.iloc[idx].reset_index(drop=True)
    X_bg = X.iloc[bg_idx]

    X_sample_scaled = torch.tensor(
        scaler.transform(X_sample.values), dtype=torch.float32, device=device
    )
    X_bg_scaled = torch.tensor(
        scaler.transform(X_bg.values), dtype=torch.float32, device=device
    )

    model.eval()
    explainer = shap.GradientExplainer(model, X_bg_scaled)
    shap_values = explainer.shap_values(X_sample_scaled)

    # shap_values may be a list of arrays (one per class) or a 3D array
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    return shap_values, X_sample
