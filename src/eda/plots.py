from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_NON_TRAINING_COLS = {"class", "class_weight", "tau_n", "eventOrigin"}


def _resolve_class_names(df: pd.DataFrame) -> list[str]:
    """Derive ordered class names from eventOrigin, falling back to integer class labels."""
    if "eventOrigin" in df.columns:
        return df.groupby("class")["eventOrigin"].first().sort_index().tolist()
    return [str(i) for i in sorted(df["class"].unique())]


def plot_class_balance(
    df: pd.DataFrame,
    class_names: list[str] | None = None,
) -> plt.Figure:
    """Plot unweighted and class-weighted event counts per class side by side."""
    class_order = sorted(df["class"].unique())
    if class_names is None:
        class_names = _resolve_class_names(df)

    n = len(class_order)
    colors = [plt.cm.tab10(i % 10) for i in range(n)]
    has_weights = "class_weight" in df.columns
    n_panels = 2 if has_weights else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    axes = np.array(axes).reshape(-1)

    counts = df["class"].value_counts().reindex(class_order)
    axes[0].bar(range(n), counts.values, color=colors)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Event count")
    axes[0].set_title("Unweighted class balance")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if has_weights:
        weighted = df.groupby("class")["class_weight"].sum().reindex(class_order)
        axes[1].bar(range(n), weighted.values, color=colors)
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Weighted event count")
        axes[1].set_title("Class-weighted balance")
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: list[str] | None = None,
    exclude: list[str] | None = None,
) -> plt.Figure:
    """Plot a Pearson correlation heatmap for numeric training features."""
    excluded = _NON_TRAINING_COLS | set(exclude or [])
    if features is None:
        features = [
            c for c in df.select_dtypes(include="number").columns if c not in excluded
        ]

    corr = df[features].corr()
    n = len(features)
    annotate = n <= 30
    size = max(8, n * 0.4)

    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annotate,
        fmt=".2f" if annotate else "",
        linewidths=0.3 if annotate else 0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str],
    class_names: list[str] | None = None,
    n_cols: int = 3,
    n_bins: int = 50,
) -> plt.Figure:
    """Plot per-class normalized histograms for each feature in a grid layout."""
    class_order = sorted(df["class"].unique())
    if class_names is None:
        class_names = _resolve_class_names(df)

    colors = [plt.cm.tab10(i % 10) for i in range(len(class_order))]
    n = len(features)
    n_rows = max(1, (n + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, feature in zip(axes_flat, features):
        for cls, name, color in zip(class_order, class_names, colors):
            values = df.loc[df["class"] == cls, feature].dropna()
            ax.hist(
                values,
                bins=n_bins,
                density=True,
                alpha=0.6,
                color=color,
                label=name,
                histtype="stepfilled",
            )
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Feature distributions by class")
    fig.tight_layout()
    return fig
