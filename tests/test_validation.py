"""Tests for Pandera MC DataFrame validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.processing.validation import validate_mc


def _make_valid_mc(n: int = 20) -> pd.DataFrame:
    """Create a minimal valid MC DataFrame."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "class": np.array([0, 1] * (n // 2), dtype=int),
            "class_weight": np.ones(n, dtype=np.float64),
            "eventOrigin": rng.randint(0, 5, size=n),
            "tau_n": rng.randint(1, 4, size=n),
            "weight": rng.uniform(0.5, 2.0, size=n).astype(np.float64),
            "feat_a": rng.randn(n).astype(np.float64),
            "feat_b": rng.randn(n).astype(np.float64),
        }
    )


class TestValidMC:
    def test_valid_passes(self) -> None:
        df = _make_valid_mc()
        result = validate_mc(df)
        assert len(result) == len(df)

    def test_extra_columns_allowed(self) -> None:
        df = _make_valid_mc()
        df["extra_col"] = 1.0
        result = validate_mc(df)
        assert "extra_col" in result.columns

    def test_three_classes(self) -> None:
        df = _make_valid_mc(30)
        df["class"] = np.tile([0, 1, 2], 10).astype(int)
        result = validate_mc(df)
        assert result["class"].nunique() == 3


class TestMissingColumns:
    def test_missing_class(self) -> None:
        df = _make_valid_mc()
        df = df.drop(columns=["class"])
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)

    def test_missing_weight(self) -> None:
        df = _make_valid_mc()
        df = df.drop(columns=["weight"])
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)

    def test_missing_class_weight(self) -> None:
        df = _make_valid_mc()
        df = df.drop(columns=["class_weight"])
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)


class TestNullValues:
    def test_null_in_training_feature(self) -> None:
        df = _make_valid_mc()
        df.loc[0, "feat_a"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)

    def test_null_in_weight(self) -> None:
        df = _make_valid_mc()
        df.loc[0, "weight"] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)


class TestDtypeChecks:
    def test_class_weight_not_positive(self) -> None:
        df = _make_valid_mc()
        df.loc[0, "class_weight"] = 0.0
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)

    def test_negative_class_label(self) -> None:
        df = _make_valid_mc()
        df.loc[0, "class"] = -1
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)

    def test_int_feature_accepted(self) -> None:
        df = _make_valid_mc()
        df["jet_n"] = np.array([3, 2] * 10, dtype=np.int64)
        result = validate_mc(df)
        assert "jet_n" in result.columns

    def test_string_feature_rejected(self) -> None:
        df = _make_valid_mc()
        df["bad_col"] = "text"
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)


class TestDataFrameChecks:
    def test_non_contiguous_classes(self) -> None:
        df = _make_valid_mc()
        df["class"] = np.where(df["class"] == 1, 3, 0)
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)

    def test_no_training_features(self) -> None:
        df = _make_valid_mc()
        df = df[["class", "class_weight", "eventOrigin", "tau_n", "weight"]]
        with pytest.raises(pa.errors.SchemaError):
            validate_mc(df)
