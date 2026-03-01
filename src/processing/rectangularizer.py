from __future__ import annotations

from typing import Literal

import awkward as ak
import pandas as pd


def rectangularize_pad_array(
    array_in: ak.Array,
    padding_threshold: int,
    nan_threshold: float = 0.0,
) -> pd.DataFrame:
    """Pad jagged features to a fixed depth and return a rectangular pandas DataFrame."""
    # Detect jagged (variable-length) features
    jagged_features = []
    for feature in ak.fields(array_in):
        try:
            ak.flatten(array_in[feature], axis=1)
            jagged_features.append(feature)
        except Exception:
            pass

    # Pad each jagged feature to padding_threshold and expand into indexed columns
    for feature in jagged_features:
        padded = ak.pad_none(array_in[feature], target=padding_threshold, clip=True)
        for i in range(padding_threshold):
            array_in = ak.with_field(array_in, padded[:, i], f"{feature}_{i}")
        array_in = array_in[[f for f in ak.fields(array_in) if f != feature]]

    # Convert to pandas DataFrame
    df = ak.to_dataframe(array_in)

    # Drop constant columns
    df = df.loc[:, df.nunique() > 1]

    # Drop columns with fewer than nan_threshold fraction of non-NaN values
    df = df.dropna(axis=1, thresh=int(nan_threshold * len(df)))

    return df


def fill_padding(
    df: pd.DataFrame,
    strategy: Literal["NaN", "0", "-999", "mean"] = "0",
) -> pd.DataFrame:
    """Replace NaN padding values in a DataFrame using the specified strategy."""
    if strategy == "NaN":
        return df
    if strategy == "0":
        return df.fillna(0)
    if strategy == "-999":
        return df.fillna(-999)
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    raise ValueError(
        f"Unsupported strategy '{strategy}'. Use 'NaN', '0', '-999', or 'mean'."
    )
