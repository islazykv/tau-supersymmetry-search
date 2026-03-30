"""ML-based regions analysis pipeline."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pyrootutils
from omegaconf import DictConfig, OmegaConf

from src.eda.utils import get_class_labels, get_class_names
from src.models.splits import (
    _NON_TRAINING_COLS,
    prepare_features_target,
    train_test_split,
)
from src.processing.analysis import get_output_paths
from src.processing.io import load_dataframe, save_dataframe
from src.regions.construction import (
    RegionThresholds,
    build_analysis_frame,
    split_into_regions,
)
from src.regions.plots import (
    plot_kinematic_distribution,
    plot_signal_score,
    plot_significance_grid,
)
from src.regions.significance import (
    compute_significance_grid,
    construct_grid,
)
from src.visualization.plots import save_figure

log = logging.getLogger(__name__)


def regions(cfg: DictConfig) -> None:
    """Run the full ML-based regions analysis pipeline."""
    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])

    log.info("Starting regions analysis:\n%s", OmegaConf.to_yaml(cfg))

    output_paths = get_output_paths(cfg)
    dataframes_dir = root / output_paths["dataframes_dir"]
    plots_dir = root / output_paths["plots_dir"] / "ml_regions"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_mc = load_dataframe(dataframes_dir / "mc.parquet")
    log.info("Loaded MC: %d events, %d columns", len(df_mc), len(df_mc.columns))

    display_labels = OmegaConf.to_container(cfg.merge.display_labels, resolve=True)
    class_names = get_class_names(df_mc)
    class_labels = get_class_labels(df_mc, display_labels=display_labels)
    log.info("Classes (%d): %s", len(class_names), class_names)

    model_type = cfg.ml_regions.model_type
    predictions_df = load_dataframe(
        dataframes_dir / f"{model_type}_predictions.parquet"
    )
    log.info("Predictions loaded: %d events", len(predictions_df))

    X, y, weights = prepare_features_target(df_mc)
    split_strategy = cfg.pipeline.split_strategy

    if split_strategy == "train_test":
        _, X_test, _, _, _, _ = train_test_split(
            X,
            y,
            weights,
            test_size=cfg.data.test_split,
            seed=cfg.seed,
        )
        test_indices = X_test.index
    elif split_strategy == "k_fold":
        # K-fold predictions are out-of-fold: each event's prediction comes from
        # a model that never saw it during training, so the full index is valid.
        test_indices = X.index
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

    analysis_df = build_analysis_frame(df_mc, predictions_df, test_indices)
    log.info(
        "Analysis frame: %d events, %d columns",
        len(analysis_df),
        len(analysis_df.columns),
    )

    t = cfg.ml_regions.thresholds
    thresholds = RegionThresholds(
        control=t.control,
        validation=t.validation,
        signal=t.signal,
    )
    regions_dict = split_into_regions(analysis_df, thresholds)

    score_cfg = cfg.ml_regions.score_plots
    threshold_tuple = (t.control, t.validation, t.signal)

    for mode in score_cfg.modes:
        for scale in score_cfg.scales:
            for weighted in score_cfg.weighted:
                fig = plot_signal_score(
                    analysis_df,
                    class_labels,
                    thresholds=threshold_tuple,
                    mode=mode,
                    weighted=weighted,
                    scale=scale,
                    bins=score_cfg.bins,
                )
                tag = f"score_{mode}_{scale}_{'w' if weighted else 'uw'}"
                save_figure(fig, plots_dir / f"{tag}.png")
                plt.close(fig)
    log.info("Signal score distributions saved.")

    background_names = list(
        OmegaConf.to_container(cfg.merge.primary_groups, resolve=True).keys()
    )
    sig_results = compute_significance_grid(
        sr_df=regions_dict["SR"],
        signal_threshold=t.signal,
        background_names=background_names,
        bins=cfg.ml_regions.significance.bins,
        bkg_uncertainty_frac=cfg.ml_regions.significance.bkg_uncertainty,
    )

    grids = {}
    for prefix, (cls_vals, m1, m2) in sig_results.items():
        grid, x_ticks, y_ticks = construct_grid(m1, m2, cls_vals)
        grids[prefix] = (grid, x_ticks, y_ticks)

    if grids:
        fig = plot_significance_grid(grids, run_label=cfg.analysis.run)
        save_figure(fig, plots_dir / "significance_grid.png")
        plt.close(fig)
        log.info("Significance grid saved.")

    training_features = [
        c
        for c in df_mc.columns
        if c not in _NON_TRAINING_COLS and df_mc[c].dtype.kind in ("i", "u", "f")
    ]
    background_display = OmegaConf.to_container(cfg.merge.display_labels, resolve=True)
    kin_cfg = cfg.ml_regions.kinematic_plots
    include_signal = kin_cfg.include_signal

    for region_name, region_df in regions_dict.items():
        bkg_dfs = {}
        for i, bkg_name in enumerate(background_names):
            mask = region_df["y_true"] == i
            if mask.any():
                bkg_dfs[background_display.get(bkg_name, bkg_name)] = region_df[mask]

        sig_dfs = {}
        if include_signal:
            for origin in kin_cfg.selected_signals:
                mask = region_df["eventOrigin"] == origin
                if mask.any():
                    sig_dfs[origin] = region_df[mask]

        for feat in training_features:
            vals = region_df[feat].dropna()
            if len(vals) == 0:
                continue

            lo, hi = float(vals.quantile(0.01)), float(vals.quantile(0.99))
            if lo == hi:
                continue

            fig = plot_kinematic_distribution(
                feature=feat,
                n_bins=40,
                xlim=(lo, hi),
                xlabel=feat,
                ylabel="Events",
                region=region_name,
                backgrounds=bkg_dfs,
                signals=sig_dfs if include_signal else None,
                run=cfg.analysis.run,
                channel=(
                    str(cfg.analysis.channel)
                    if hasattr(cfg.analysis, "channel")
                    else None
                ),
                subject=model_type.upper(),
                scale=kin_cfg.scale,
                scale_factor=kin_cfg.scale_factor,
            )
            save_figure(fig, plots_dir / f"kin_{region_name}_{feat}.png")
            plt.close(fig)

        log.info("Kinematic distributions for %s saved.", region_name)

    for region_name, region_df in regions_dict.items():
        save_dataframe(
            region_df,
            dataframes_dir / f"{model_type}_{region_name.lower()}.parquet",
        )

    log.info("Regions analysis complete — plots saved to %s", plots_dir)
