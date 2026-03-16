"""CLs significance computation for signal mass points in the Signal Region."""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import pyhf
from tqdm import tqdm

log = logging.getLogger(__name__)


def compute_significance_grid(
    sr_df: pd.DataFrame,
    signal_threshold: float,
    background_names: list[str],
    bins: int = 20,
    bkg_uncertainty_frac: float = 0.3,
    score_column: str = "p_signal",
) -> dict[str, tuple[list[float], list[int], list[int]]]:
    """Compute expected CLs for each signal mass point in the Signal Region.

    Uses pyhf's uncorrelated-background model with the q-tilde test statistic.

    Parameters
    ----------
    sr_df : pd.DataFrame
        Signal Region DataFrame with ``eventOrigin``, ``weight``, and *score_column*.
    signal_threshold : float
        Lower edge of the signal score binning (upper edge is 1).
    background_names : list[str]
        Names of background classes as they appear in ``eventOrigin``.
    bins : int
        Number of score bins between *signal_threshold* and 1.
    bkg_uncertainty_frac : float
        Fractional systematic uncertainty on the background yield per bin.
    score_column : str
        Column containing the signal probability score.

    Returns
    -------
    dict mapping signal type prefix (e.g. ``"GG"``, ``"SS"``) to a tuple of
    ``(cls_values, mass1_list, mass2_list)``.
    """
    bin_edges = np.linspace(signal_threshold, 1, bins + 1)

    df_bkg = sr_df[sr_df["eventOrigin"].isin(background_names)]
    bkg_yield = np.histogram(
        df_bkg[score_column], weights=df_bkg["weight"], bins=bin_edges
    )[0]

    bkg = list(bkg_yield)
    bkg_uncert = list(bkg_yield * bkg_uncertainty_frac)

    results: dict[str, tuple[list[float], list[int], list[int]]] = {}

    pyhf.set_backend("numpy")

    mass_points = [
        mp for mp in sr_df["eventOrigin"].unique() if mp not in background_names
    ]

    for mass_point in tqdm(mass_points, desc="CLs significance"):
        sr_mp = sr_df[sr_df["eventOrigin"] == mass_point]
        signal_yield = np.histogram(
            sr_mp[score_column], weights=sr_mp["weight"], bins=bin_edges
        )[0]
        signal = list(signal_yield)

        model = pyhf.simplemodels.uncorrelated_background(
            signal=signal, bkg=bkg, bkg_uncertainty=bkg_uncert
        )
        data = bkg + model.config.auxdata

        try:
            # pyhf prints verbose optimizer diagnostics to stderr; suppress them.
            _stderr = sys.stderr
            sys.stderr = open("/dev/null", "w")  # noqa: SIM115
            _, cls_exp = pyhf.infer.hypotest(
                1.0, data, model, test_stat="qtilde", return_expected=True
            )
            sys.stderr.close()
            sys.stderr = _stderr
        except Exception:
            log.debug("pyhf hypotest failed for %s — defaulting to CLs=0.0", mass_point)
            cls_exp = 0.0

        prefix = mass_point.split("_")[0]
        if prefix not in results:
            results[prefix] = ([], [], [])

        cls_list, m1_list, m2_list = results[prefix]
        cls_list.append(float(np.round(cls_exp, 3)))

        parts = mass_point.split("_")
        m1_list.append(int(parts[1]))
        m2_list.append(int(parts[2]))

    return results


def construct_grid(
    x_list: list[int],
    y_list: list[int],
    z_list: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot significance values into a 2D grid for heatmap plotting.

    Parameters
    ----------
    x_list, y_list : list[int]
        Mass coordinates.
    z_list : list[float]
        Significance values.

    Returns
    -------
    grid : np.ndarray
        Pivoted 2D values array.
    x_ticks : np.ndarray
        Sorted unique x-axis mass values.
    y_ticks : np.ndarray
        Sorted unique y-axis mass values.
    """
    df = pd.DataFrame({"x": x_list, "y": y_list, "z": z_list})
    pivoted = df.pivot(index="y", columns="x", values="z")
    return pivoted.values, pivoted.columns.values, pivoted.index.values
