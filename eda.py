import logging

import hydra
import matplotlib
import pyrootutils
from omegaconf import DictConfig

matplotlib.use("Agg")

pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

from src.eda.checks import (  # noqa: E402
    summarize_feature_ranges,
    summarize_missing,
    summarize_zeros,
)
from src.eda.plots import (  # noqa: E402
    plot_class_balance,
    plot_correlation_matrix,
    plot_feature_distributions,
)
from src.processing.analysis import get_output_paths  # noqa: E402
from src.processing.io import load_dataframe  # noqa: E402
from src.visualization.plots import save_figure  # noqa: E402

log = logging.getLogger(__name__)

_NON_TRAINING_COLS = {"class", "class_weight", "tau_n", "eventOrigin"}


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Run the full EDA pipeline: load, check data quality, and save plots."""
    output_paths = get_output_paths(cfg)
    dataframes_dir = output_paths["dataframes_dir"]
    plots_dir = output_paths["plots_dir"] / "eda"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    df_mc = load_dataframe(dataframes_dir / "mc.parquet")
    log.info("Loaded MC: %d rows, %d columns", len(df_mc), len(df_mc.columns))

    # --- class labels ---
    class_names = df_mc.groupby("class")["eventOrigin"].first().sort_index().tolist()
    log.info("Classes: %s", class_names)

    # --- data quality ---
    missing = summarize_missing(df_mc)
    if missing.empty:
        log.info("Missing values: none")
    else:
        log.warning("Missing values detected:\n%s", missing.to_string())

    zeros = summarize_zeros(df_mc, exclude=list(_NON_TRAINING_COLS))
    log.info(
        "Zero-value summary (%d columns with zeros):\n%s", len(zeros), zeros.to_string()
    )

    ranges = summarize_feature_ranges(df_mc)
    log.info(
        "Feature ranges computed for %d features",
        len(ranges.columns.get_level_values(0).unique()),
    )

    # --- plots ---
    log.info("Generating class balance plot...")
    fig = plot_class_balance(df_mc, class_names=class_names)
    save_figure(fig, plots_dir / "class_balance.png")

    log.info("Generating correlation matrix...")
    fig = plot_correlation_matrix(df_mc)
    save_figure(fig, plots_dir / "correlation_matrix.png")

    log.info("Generating feature distributions...")
    training_cols = [
        c
        for c in df_mc.select_dtypes(include="number").columns
        if c not in _NON_TRAINING_COLS
    ]
    fig = plot_feature_distributions(
        df_mc, features=training_cols[:12], class_names=class_names
    )
    save_figure(fig, plots_dir / "feature_distributions.png")

    log.info("EDA complete — plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
