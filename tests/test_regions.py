"""Tests for region construction and significance grid utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regions.construction import (
    RegionThresholds,
    build_analysis_frame,
    split_into_regions,
)
from src.regions.significance import construct_grid


class TestSplitIntoRegions:
    def test_boundaries(self) -> None:
        df = pd.DataFrame({"p_signal": [0.1, 0.3, 0.5, 0.7, 0.9]})
        thresholds = RegionThresholds(control=0.2, validation=0.4, signal=0.6)
        regions = split_into_regions(df, thresholds)

        assert len(regions["CR"]) == 1  # 0.3
        assert len(regions["VR"]) == 1  # 0.5
        assert len(regions["SR"]) == 2  # 0.7, 0.9

    def test_no_overlap(self) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"p_signal": rng.uniform(0, 1, size=200)})
        thresholds = RegionThresholds(control=0.2, validation=0.5, signal=0.8)
        regions = split_into_regions(df, thresholds)

        all_indices = set()
        for region_df in regions.values():
            idx = set(region_df.index)
            assert all_indices.isdisjoint(idx), "Overlap between regions"
            all_indices.update(idx)

    def test_empty_sr(self) -> None:
        df = pd.DataFrame({"p_signal": [0.1, 0.2, 0.3]})
        thresholds = RegionThresholds(control=0.0, validation=0.5, signal=0.99)
        regions = split_into_regions(df, thresholds)
        assert len(regions["SR"]) == 0


class TestBuildAnalysisFrame:
    def test_happy_path(self) -> None:
        df_mc = pd.DataFrame(
            {
                "eventOrigin": ["a", "b", "c", "d"],
                "weight": [1.0, 1.0, 1.0, 1.0],
                "feat": [10, 20, 30, 40],
            }
        )
        preds = pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]})
        test_indices = pd.Index([1, 3])

        result = build_analysis_frame(df_mc, preds, test_indices)
        assert len(result) == 2
        assert "eventOrigin" in result.columns
        assert "y_pred" in result.columns

    def test_length_mismatch_raises(self) -> None:
        df_mc = pd.DataFrame({"feat": [1, 2, 3]})
        preds = pd.DataFrame({"y_pred": [0, 1]})  # 2 rows vs 3
        with pytest.raises(ValueError, match="different lengths"):
            build_analysis_frame(df_mc, preds, df_mc.index)


class TestConstructGrid:
    def test_pivot_shape(self) -> None:
        x = [100, 200, 100, 200]
        y = [50, 50, 100, 100]
        z = [0.1, 0.2, 0.3, 0.4]
        grid, x_ticks, y_ticks = construct_grid(x, y, z)
        assert grid.shape == (2, 2)
        assert len(x_ticks) == 2
        assert len(y_ticks) == 2

    def test_pivot_values(self) -> None:
        x = [100, 200]
        y = [50, 50]
        z = [0.5, 0.9]
        grid, x_ticks, y_ticks = construct_grid(x, y, z)
        assert grid[0, 0] == 0.5
        assert grid[0, 1] == 0.9
