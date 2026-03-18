from __future__ import annotations

import awkward as ak
import pandas as pd
import pytest

from src.processing.rectangularizer import fill_padding, rectangularize_pad_array


def test_pad_array_expands_jagged_features():
    array = ak.Array(
        {
            "met": [100.0, 200.0, 300.0],
            "jet_pt": [[30.0, 40.0], [50.0], [60.0, 70.0, 80.0]],
        }
    )
    df = rectangularize_pad_array(array, padding_threshold=2)

    assert "jet_pt_0" in df.columns
    assert "jet_pt_1" in df.columns
    assert "jet_pt" not in df.columns
    assert len(df) == 3


def test_pad_array_drops_constant_columns():
    array = ak.Array(
        {
            "met": [100.0, 200.0, 300.0],
            "constant": [1.0, 1.0, 1.0],
        }
    )
    df = rectangularize_pad_array(array, padding_threshold=1)

    assert "constant" not in df.columns
    assert "met" in df.columns


def test_pad_array_nan_threshold_drops_sparse_columns():
    array = ak.Array(
        {
            "met": [100.0, 200.0, 300.0],
            "jet_pt": [[30.0], [], [60.0]],
        }
    )
    df_all = rectangularize_pad_array(array, padding_threshold=2, nan_threshold=0.0)
    df_strict = rectangularize_pad_array(array, padding_threshold=2, nan_threshold=1.0)

    assert len(df_all.columns) >= len(df_strict.columns)


def _make_df_with_nans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, float("nan"), 3.0],
            "b": [float("nan"), 2.0, float("nan")],
        }
    )


def test_fill_padding_zero():
    result = fill_padding(_make_df_with_nans(), strategy="0")

    assert result.isna().sum().sum() == 0
    assert result["a"].iloc[1] == 0.0


def test_fill_padding_nan_keeps_nans():
    result = fill_padding(_make_df_with_nans(), strategy="NaN")

    assert result.isna().sum().sum() == 3


def test_fill_padding_negative_999():
    result = fill_padding(_make_df_with_nans(), strategy="-999")

    assert result["a"].iloc[1] == -999
    assert result["b"].iloc[0] == -999


def test_fill_padding_mean():
    result = fill_padding(_make_df_with_nans(), strategy="mean")

    assert result["a"].iloc[1] == pytest.approx(2.0)
    assert result.isna().sum().sum() == 0


def test_fill_padding_invalid_strategy():
    with pytest.raises(ValueError, match="Unsupported strategy"):
        fill_padding(_make_df_with_nans(), strategy="invalid")
