from __future__ import annotations

import pandas as pd


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and fractions of missing values for columns that have any."""
    counts = df.isnull().sum()
    result = pd.DataFrame({"missing": counts, "fraction": counts / len(df)})
    return result[result["missing"] > 0].sort_values("fraction", ascending=False)


def summarize_zeros(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Return counts and fractions of zero values per numeric column, excluding specified columns."""
    excluded = set(exclude or [])
    cols = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
    counts = (df[cols] == 0).sum()
    result = pd.DataFrame({"zeros": counts, "fraction": counts / len(df)})
    return result[result["zeros"] > 0].sort_values("fraction", ascending=False)


def summarize_feature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-class min, max, mean and std statistics for all numeric features."""
    numeric = df.select_dtypes(include="number")
    return numeric.groupby(df["class"]).agg(["min", "max", "mean", "std"])
