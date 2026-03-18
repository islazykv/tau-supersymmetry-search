"""Plotting functions for ML-based regions analysis."""

from __future__ import annotations

import atlas_mpl_style as ampl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_LEGEND_KW: dict = dict(
    loc="upper right",
    prop={"size": 12},
    labelspacing=0.1,
    handlelength=1,
    handleheight=1,
)

_REGION_COLORS: dict[str, str] = {"CR": "red", "VR": "blue", "SR": "green"}


def plot_signal_score(
    df: pd.DataFrame,
    class_labels: list[str],
    *,
    thresholds: tuple[float, float, float] | None = None,
    mode: str = "binary",
    weighted: bool = False,
    scale: str = "linear",
    bins: int = 50,
    score_column: str = "p_signal",
) -> plt.Figure:
    """Stacked histogram of the signal output score with region threshold overlays."""
    scores = df[score_column].to_numpy()
    y_true = df["y_true"].to_numpy()
    weights = df["weight"].to_numpy()
    n_classes = len(class_labels)

    bin_edges = np.linspace(0, 1, bins + 1)
    bin_size = 1 / bins

    if mode == "binary":
        bkg_mask = y_true < n_classes - 1
        sig_mask = y_true == n_classes - 1
        classes_array = [scores[bkg_mask], scores[sig_mask]]
        weights_array = [weights[bkg_mask], weights[sig_mask]]
        labels = ["background", "signal"]
        n_colors = 2
    elif mode == "multiclass":
        classes_array = [scores[y_true == i] for i in range(n_classes)]
        weights_array = [weights[y_true == i] for i in range(n_classes)]
        labels = list(class_labels)
        n_colors = n_classes
    else:
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'binary' or 'multiclass'.")

    color = list(sns.color_palette("Set3", n_colors=n_colors))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    hist_kwargs: dict = dict(
        bins=bin_edges,
        histtype="bar",
        stacked=True,
        label=labels,
        color=color[::-1],
    )
    if weighted:
        hist_kwargs["weights"] = weights_array

    ax.hist(classes_array, **hist_kwargs)

    if thresholds is not None:
        t_cr, t_vr, t_sr = thresholds
        ax.axvline(t_cr, color="red", label=f"CR threshold: {t_cr}")
        ax.axvline(t_vr, color="blue", label=f"VR threshold: {t_vr}")
        ax.axvline(t_sr, color="green", label=f"SR threshold: {t_sr}")
        ax.axvspan(t_cr, t_vr, facecolor="red", alpha=0.2)
        ax.axvspan(t_vr, t_sr, facecolor="blue", alpha=0.2)
        ax.axvspan(t_sr, 1, facecolor="green", alpha=0.2)
        ax.text(
            (t_cr + t_vr) / 2,
            0.5,
            "CR",
            fontsize=20,
            ha="center",
            va="center",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            (t_vr + t_sr) / 2,
            0.5,
            "VR",
            fontsize=20,
            ha="center",
            va="center",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            t_sr + (1 - t_sr) / 2,
            0.5,
            "SR",
            fontsize=20,
            ha="center",
            va="center",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlabel("Signal Output Score", fontsize=12, loc="right")
    ax.set_ylabel(f"Events / {bin_size:.2g}", fontsize=12, loc="top")
    ax.set_xlim([0, 1])

    ymin, ymax = ax.get_ylim()
    if scale == "linear":
        ax.set_ylim([0, ymax * 1.3])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 8))
    elif scale == "log":
        ax.set_yscale("log")
        ax.set_ylim([1, ymax * 10])
    else:
        raise ValueError(f"Unsupported scale: {scale!r}.")

    ax.locator_params(nbins=10, axis="x")
    if scale == "linear":
        ax.locator_params(nbins=10, axis="y")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(**_LEGEND_KW)
    ampl.draw_atlas_label(0.02, 0.97, status="int")

    return fig


def plot_significance_grid(
    grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    run_label: str = "Run2",
) -> plt.Figure:
    """Side-by-side heatmaps of CLs significance for each signal production type."""
    _TITLES: dict[str, str] = {
        "GG": (
            r"$\tilde{g}\tilde{g}$ production, "
            r"$\tilde{g} \to q q \tau \nu {\tilde{\chi}^{0}_{1}}$ / "
            r"$q q \tau \tau {\tilde{\chi}^{0}_{1}}$ / "
            r"$q q \nu \nu {\tilde{\chi}^{0}_{1}}$"
        ),
        "SS": (
            r"$\tilde{s}\tilde{s}$ production, "
            r"$\tilde{s} \to q \tau \nu {\tilde{\chi}^{0}_{1}}$ / "
            r"$q \tau \tau {\tilde{\chi}^{0}_{1}}$ / "
            r"$q \nu \nu {\tilde{\chi}^{0}_{1}}$"
        ),
    }

    _XLABELS: dict[str, str] = {
        "GG": r"$m_{\tilde{g}}$ [GeV]",
        "SS": r"$m_{\tilde{s}}$ [GeV]",
    }

    if run_label == "Run2":
        lumi_text = r"$\sqrt{s}=13$" + " TeV, 140.1 fb" + r"$^{-1}$"
    elif run_label == "Run3":
        lumi_text = r"$\sqrt{s}=13.6$" + " TeV, 51.8 fb" + r"$^{-1}$"
    else:
        lumi_text = run_label

    n_panels = len(grids)
    fig, axes = plt.subplots(1, n_panels, figsize=(17 * n_panels, 12))
    if n_panels == 1:
        axes = [axes]

    for ax, (key, (grid, x_ticks, y_ticks)) in zip(axes, grids.items()):
        sns.heatmap(
            grid[::-1],
            ax=ax,
            xticklabels=x_ticks,
            yticklabels=y_ticks[::-1],
            cmap="coolwarm",
            square=False,
            annot=True,
            fmt=".2f",
            cbar_kws={"ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            vmin=0,
            vmax=1,
        )
        ax.set_title(_TITLES.get(key, key), loc="left")
        ax.set_xlabel(_XLABELS.get(key, "mass [GeV]"), loc="right", fontsize=32)
        ax.set_ylabel(r"$m_{\tilde{\chi}^{0}_{1}}$ [GeV]", loc="top", fontsize=32)
        ax.text(0.22, 0.95, lumi_text, transform=ax.transAxes)
        ampl.draw_atlas_label(0.02, 0.97, status="int", ax=ax)

    fig.tight_layout()
    return fig


def _build_caption(
    run: str,
    region: str,
    channel: str | None = None,
    subject: str | None = None,
) -> str:
    """Build the ATLAS-style caption text block."""
    if run == "Run2":
        upper = r"$\sqrt{s}=13$" + " TeV, 140.1 fb" + r"$^{-1}$" + "\n"
    elif run == "Run3":
        upper = r"$\sqrt{s}=13.6$" + " TeV, 51.8 fb" + r"$^{-1}$" + "\n"
    else:
        upper = f"{run}\n"

    middle = f"{channel}" + r"$\tau$ channel" + "\n" if channel else "\n"

    if subject:
        lower = f"{subject} {region}"
    else:
        lower = region

    return upper + middle + lower


def _draw_kinematic_ax(
    ax_main: plt.Axes,
    feature: str,
    bin_edges: np.ndarray,
    bkg_arrays: list[np.ndarray],
    bkg_weights: list[np.ndarray],
    bkg_labels: list[str],
    color_bkg: list,
    *,
    sig_arrays: list[np.ndarray] | None = None,
    sig_weights: list[np.ndarray] | None = None,
    sig_labels: list[str] | None = None,
    color_sig: list | None = None,
    data: pd.DataFrame | None = None,
    data_label: str = "Data",
    scale: str = "linear",
    xlabel: str = "",
    ylabel: str = "Events",
    xlim: tuple[float, float] | None = None,
    ax_ratio: plt.Axes | None = None,
    run: str = "Run2",
    region: str = "",
    channel: str | None = None,
    subject: str | None = None,
) -> None:
    """Draw a single kinematic distribution panel on the given axes."""
    bins_center = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    include_data = data is not None

    ax_main.set_xlim(xlim)
    if ax_ratio is not None:
        ax_ratio.set_xlim(xlim)

    # ---- signals (step histograms) ----
    n_sig = 0
    if sig_arrays is not None and len(sig_arrays) > 0:
        n_sig = len(sig_arrays)
        ax_main.hist(
            sig_arrays,
            weights=sig_weights,
            bins=bin_edges,
            histtype="step",
            label=sig_labels[::-1],
            color=color_sig,
            linestyle="dashed",
            linewidth=2,
        )

    # ---- merged background outline (red step) ----
    all_bkg_vals = np.concatenate(bkg_arrays)
    all_bkg_w = np.concatenate(bkg_weights)
    n_bkg_merged, _, _ = ax_main.hist(
        all_bkg_vals,
        weights=all_bkg_w,
        bins=bin_edges,
        histtype="step",
        stacked=True,
        color="red",
        linewidth=2,
    )

    # ---- stacked backgrounds (filled bars) ----
    ax_main.hist(
        bkg_arrays[::-1],
        weights=bkg_weights[::-1],
        bins=bin_edges,
        histtype="bar",
        stacked=True,
        label=bkg_labels[::-1],
        color=color_bkg,
    )

    # ---- data points ----
    n_data_safe = None
    if include_data:
        data_vals = data[feature].to_numpy()
        data_w = data["weight"].to_numpy()
        n_data, _ = np.histogram(data_vals, weights=data_w, bins=bin_edges)
        n_data_safe = np.where(n_data == 0, np.nan, n_data)
        ax_main.scatter(
            bins_center,
            n_data_safe,
            marker="o",
            c="black",
            s=40,
            label=data_label,
            zorder=2,
        )

    # ---- ratio panel ----
    if include_data and ax_ratio is not None:
        assert n_data_safe is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = n_data_safe / n_bkg_merged
            ratio_err = np.sqrt(n_data_safe) / n_data_safe
        ax_ratio.scatter(bins_center, ratio, marker="o", c="black", s=40, zorder=2)
        ax_ratio.errorbar(bins_center, ratio, yerr=ratio_err, fmt=".", c="black")
        ax_ratio.set_ylim([0.7, 1.3])
        ax_ratio.set_ylabel("Data / SM", fontsize=12)
        ax_ratio.set_xlabel(xlabel, loc="right", fontsize=12)
        ax_ratio.locator_params(nbins=10, axis="x")
        ax_ratio.tick_params(axis="both", labelsize=12)
        ax_main.set_xticklabels([])
    else:
        ax_main.set_xlabel(xlabel, loc="right", fontsize=12)
        ax_main.locator_params(nbins=10, axis="x")
        ax_main.tick_params(axis="x", labelsize=12)

    # ---- y-axis ----
    ax_main.set_ylabel(ylabel, loc="top", fontsize=12)
    ax_main.tick_params(axis="y", labelsize=12)

    ymin, ymax = ax_main.get_ylim()
    if scale == "linear":
        ax_main.set_ylim([0, ymax * 1.3])
        ax_main.ticklabel_format(axis="y", style="sci", scilimits=(0, 7))
        ax_main.locator_params(nbins=10, axis="y")
    elif scale == "log":
        ax_main.set_yscale("log")
        ax_main.set_ylim([1, ymax * 10])
    else:
        raise ValueError(f"Unsupported scale: {scale!r}.")

    # ---- legend ----
    handles, labels = ax_main.get_legend_handles_labels()

    sig_handles = handles[:n_sig]
    sig_legend_labels = labels[:n_sig]
    bkg_handles = handles[n_sig:]
    bkg_legend_labels = labels[n_sig:]

    first_legend_handles = []
    first_legend_labels = []

    if include_data:
        first_legend_handles.append(
            mpatches.Patch(facecolor="red", alpha=0.3, edgecolor="red")
        )
        first_legend_labels.append(r"SM $\pm 1 \sigma$")

    if n_sig > 0:
        first_legend_handles.extend(sig_handles)
        first_legend_labels.extend(sig_legend_labels)

    if first_legend_handles:
        first_legend = ax_main.legend(
            first_legend_handles,
            first_legend_labels,
            bbox_to_anchor=[0.21, 0.8, 1, 0.2],
            loc="upper center",
            **{k: v for k, v in _LEGEND_KW.items() if k != "loc"},
        )
        ax_main.add_artist(first_legend)

    ax_main.legend(
        bkg_handles[::-1],
        bkg_legend_labels[::-1],
        **_LEGEND_KW,
    )

    # ---- ATLAS label & caption ----
    ampl.draw_atlas_label(0.02, 0.97, status="int", ax=ax_main)
    caption = _build_caption(run, region, channel, subject)
    ax_main.text(0.33, 0.85, caption, fontsize=12, transform=ax_main.transAxes)


def plot_kinematic_distribution(
    feature: str,
    n_bins: int,
    xlim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    region: str,
    backgrounds: dict[str, pd.DataFrame],
    *,
    signals: dict[str, pd.DataFrame] | None = None,
    background_labels: list[str] | None = None,
    signal_labels: list[str] | None = None,
    data: pd.DataFrame | None = None,
    data_label: str = "Data",
    run: str = "Run2",
    channel: str | None = None,
    subject: str | None = None,
    scale: str = "linear",
    scale_factor: float = 1.0,
) -> plt.Figure:
    """Stacked background + signal overlay kinematic distribution (supports dual linear/log panels)."""
    include_data = data is not None
    bin_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)

    bkg_keys = list(backgrounds.keys())
    bkg_labels_final = background_labels if background_labels is not None else bkg_keys
    n_bkg = len(bkg_keys)
    color_bkg = list(sns.color_palette("Set3", n_colors=n_bkg))

    # ---- extract arrays ----
    bkg_arrays = [backgrounds[k][feature].to_numpy() for k in bkg_keys]
    bkg_weights = [backgrounds[k]["weight"].to_numpy() * scale_factor for k in bkg_keys]

    sig_arrays = sig_weights = sig_labels_final = color_sig = None
    if signals:
        sig_keys = list(signals.keys())
        sig_labels_final = signal_labels if signal_labels is not None else sig_keys
        color_sig = list(sns.color_palette("rocket", n_colors=len(sig_keys)))
        sig_arrays = [signals[k][feature].to_numpy() for k in sig_keys]
        sig_weights = [signals[k]["weight"].to_numpy() for k in sig_keys]

    dual = scale == "dual"
    scales = ["linear", "log"] if dual else [scale]
    n_cols = len(scales)

    # ---- figure layout ----
    if include_data:
        fig, axes = plt.subplots(
            2,
            n_cols,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(8 * n_cols, 6),
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(8 * n_cols, 6),
            squeeze=False,
        )

    for col_idx, s in enumerate(scales):
        ax_main = axes[0, col_idx]
        ax_ratio = axes[1, col_idx] if include_data else None

        _draw_kinematic_ax(
            ax_main,
            feature,
            bin_edges,
            bkg_arrays,
            bkg_weights,
            bkg_labels_final,
            color_bkg,
            sig_arrays=sig_arrays,
            sig_weights=sig_weights,
            sig_labels=sig_labels_final,
            color_sig=color_sig,
            data=data,
            data_label=data_label,
            scale=s,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ax_ratio=ax_ratio,
            run=run,
            region=region,
            channel=channel,
            subject=subject,
        )

    if include_data:
        for col_idx in range(n_cols):
            fig.align_ylabels([axes[0, col_idx], axes[1, col_idx]])

    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    return fig
