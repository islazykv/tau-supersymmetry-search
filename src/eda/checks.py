from __future__ import annotations

import pandas as pd


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and fractions for columns that have any."""
    counts = df.isnull().sum()
    result = pd.DataFrame({"missing": counts, "fraction": counts / len(df)})
    return result[result["missing"] > 0].sort_values("fraction", ascending=False)


def summarize_feature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-class min, max, mean, and std for all numeric features."""
    numeric = df.select_dtypes(include="number")
    return numeric.groupby(df["class"]).agg(["min", "max", "mean", "std"])
