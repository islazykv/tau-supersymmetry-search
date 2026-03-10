"""ML-based region construction: split events into CR, VR, SR by signal score thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionThresholds:
    """Score boundaries defining Control, Validation, and Signal Regions."""

    control: float
    validation: float
    signal: float


def split_into_regions(
    df: pd.DataFrame,
    thresholds: RegionThresholds,
    score_column: str = "p_signal",
) -> dict[str, pd.DataFrame]:
    """Split a predictions DataFrame into CR, VR, SR based on signal score thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with predictions and metadata.
    thresholds : RegionThresholds
        Score boundaries for CR, VR, SR.
    score_column : str
        Name of the column containing the signal output score.

    Returns
    -------
    dict mapping region name ("CR", "VR", "SR") to filtered DataFrames.
    """
    score = df[score_column]
    regions = {
        "CR": df[(score >= thresholds.control) & (score < thresholds.validation)],
        "VR": df[(score >= thresholds.validation) & (score < thresholds.signal)],
        "SR": df[score >= thresholds.signal],
    }
    for name, region_df in regions.items():
        log.info("Region %s: %d events", name, len(region_df))
    return regions


def build_analysis_frame(
    df_mc: pd.DataFrame,
    predictions_df: pd.DataFrame,
    test_indices: pd.Index,
) -> pd.DataFrame:
    """Combine mc.parquet metadata/kinematics with predictions for test-set events.

    Aligns the MC metadata (eventOrigin, weight, tau_n, kinematic features) for the
    test-set rows with the corresponding predictions (y_true, y_pred, p_<class>).

    Parameters
    ----------
    df_mc : pd.DataFrame
        Full MC DataFrame with eventOrigin, weight, tau_n, and kinematic features.
    predictions_df : pd.DataFrame
        Predictions DataFrame with y_true, y_pred, p_<class> columns.
    test_indices : pd.Index
        The row indices from df_mc corresponding to the test set.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame indexed by test events.
    """
    mc_test = df_mc.loc[test_indices].reset_index(drop=True)
    preds = predictions_df.reset_index(drop=True)

    if len(mc_test) != len(preds):
        raise ValueError(
            f"MC test set ({len(mc_test)}) and predictions ({len(preds)}) "
            "have different lengths — check split strategy and seed."
        )

    return pd.concat([mc_test, preds], axis=1)
