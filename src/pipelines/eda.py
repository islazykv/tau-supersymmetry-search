"""Exploratory data analysis pipeline: load, check, and plot."""

from __future__ import annotations

import logging

import pyrootutils
from omegaconf import DictConfig, OmegaConf

from src.eda.checks import (
    summarize_feature_ranges,
    summarize_missing,
)
from src.eda.plots import (
    plot_class_balance,
    plot_correlation_matrix,
    plot_feature_distributions,
)
from src.eda.utils import get_class_labels, get_class_names
from src.processing.analysis import get_output_paths
from src.processing.io import load_dataframe
from src.processing.validation import METADATA_COLUMNS
from src.visualization.plots import save_figure

log = logging.getLogger(__name__)


def eda(cfg: DictConfig) -> None:
    """Run the full EDA pipeline: load, check data quality, and save plots."""
    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    dataframes_dir = root / output_paths["dataframes_dir"]
    plots_dir = root / output_paths["plots_dir"] / "eda"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_mc = load_dataframe(dataframes_dir / "mc.parquet")
    log.info("Loaded MC: %d rows, %d columns", len(df_mc), len(df_mc.columns))

    display_labels = OmegaConf.to_container(cfg.merge.display_labels, resolve=True)
    class_names = get_class_names(df_mc)
    class_labels = get_class_labels(df_mc, display_labels=display_labels)
    log.info("Classes: %s", class_names)

    missing = summarize_missing(df_mc)
    if missing.empty:
        log.info("Missing values: none")
    else:
        log.warning("Missing values detected:\n%s", missing.to_string())

    ranges = summarize_feature_ranges(df_mc)
    log.info(
        "Feature ranges computed for %d features",
        len(ranges.columns.get_level_values(0).unique()),
    )

    log.info("Generating class balance plot...")
    fig = plot_class_balance(df_mc, class_labels=class_labels)
    save_figure(fig, plots_dir / "class_balance.png")

    log.info("Generating correlation matrix...")
    fig = plot_correlation_matrix(df_mc)
    save_figure(fig, plots_dir / "correlation_matrix.png")

    log.info("Generating feature distributions...")
    training_cols = [
        c
        for c in df_mc.select_dtypes(include="number").columns
        if c not in METADATA_COLUMNS
    ]
    fig = plot_feature_distributions(
        df_mc, features=training_cols[:12], class_labels=class_labels
    )
    save_figure(fig, plots_dir / "feature_distributions.png")

    log.info("EDA complete — plots saved to %s", plots_dir)
