from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.processing.merger import (
    assign_class,
    combine_samples,
    dict_to_array,
    group_samples,
    merge_backgrounds,
    merge_signals,
    split_mc_data,
)


def _make_array(n: int, seed: int = 0) -> ak.Array:
    """Create a small awkward array with random 'met' and 'weight' fields.

    Args:
        n: Number of events.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    return ak.Array({"met": rng.random(n), "weight": rng.random(n)})


def test_merge_backgrounds_as_one():
    samples = {"ttbar": _make_array(5), "diboson": _make_array(3)}
    cfg = OmegaConf.create({"merge": {"background_strategy": "as_one"}})
    result = merge_backgrounds(samples, cfg)
    assert list(result.keys()) == ["background"]
    assert len(result["background"]) == 8


def test_merge_backgrounds_as_primary():
    samples = {
        "ttbar": _make_array(5),
        "singletop": _make_array(3),
        "diboson": _make_array(4),
    }
    cfg = OmegaConf.create(
        {
            "merge": {
                "background_strategy": "as_primary",
                "primary_groups": {
                    "topquarks": ["ttbar", "singletop"],
                    "diboson": ["diboson"],
                },
            }
        }
    )
    result = merge_backgrounds(samples, cfg)
    assert set(result.keys()) == {"topquarks", "diboson"}
    assert len(result["topquarks"]) == 8
    assert len(result["diboson"]) == 4


def test_combine_samples():
    bg = {"background": _make_array(5)}
    sig = {"signal_100_200": _make_array(3)}
    result = combine_samples(bg, sig)
    assert set(result.keys()) == {"background", "signal_100_200"}


def test_assign_class():
    samples = {"bg": _make_array(3), "sig": _make_array(2)}
    assign_class(samples)
    assert ak.all(samples["bg"]["class"] == 0)
    assert ak.all(samples["sig"]["class"] == 1)


def test_dict_to_array():
    samples = {"a": _make_array(3), "b": _make_array(4)}
    result = dict_to_array(samples)
    assert len(result) == 7


def test_merge_signals_as_is():
    signal = {"GG_100_200": _make_array(3), "SS_300_400": _make_array(4)}
    cfg = OmegaConf.create({"merge": {"signal_strategy": "as_is"}})
    result = merge_signals(signal, cfg)

    assert set(result.keys()) == {"GG_100_200", "SS_300_400"}
    assert len(result["GG_100_200"]) == 3


def test_merge_signals_as_one():
    signal = {"GG_100_200": _make_array(3), "SS_300_400": _make_array(4)}
    cfg = OmegaConf.create({"merge": {"signal_strategy": "as_one"}})
    result = merge_signals(signal, cfg)

    assert list(result.keys()) == ["signal"]
    assert len(result["signal"]) == 7


def test_merge_signals_as_type():
    signal = {"GG_100_200": _make_array(3), "SS_300_400": _make_array(4)}
    cfg = OmegaConf.create(
        {
            "merge": {
                "signal_strategy": "as_type",
                "signal_type_names": {"GG": "gluinos", "SS": "squarks"},
            }
        }
    )
    result = merge_signals(signal, cfg)

    assert set(result.keys()) == {"gluinos", "squarks"}
    assert len(result["gluinos"]) == 3
    assert len(result["squarks"]) == 4


def test_merge_signals_invalid_strategy():
    signal = {"GG_100_200": _make_array(3)}
    cfg = OmegaConf.create(
        {
            "merge": {
                "signal_strategy": "bad",
                "signal_type_names": {"GG": "gluinos"},
            }
        }
    )

    with pytest.raises(ValueError, match="Unsupported signal_strategy"):
        merge_signals(signal, cfg)


def test_group_samples_categorizes_correctly():
    samples = {
        "data22": _make_array(2),
        "ttbar": _make_array(3),
        "GG_100_200": _make_array(4),
    }
    cfg = OmegaConf.create(
        {
            "samples": {
                "data": {
                    "enabled": True,
                    "samples": [{"id": "data22", "label": "data"}],
                },
                "signal": {"enabled": True, "filter_patterns": ["GG_"]},
            }
        }
    )
    grouped = group_samples(samples, cfg)

    assert "data22" in grouped["data"]
    assert "ttbar" in grouped["background"]
    assert "GG_100_200" in grouped["signal"]


def test_group_samples_all_background_when_disabled():
    samples = {"ttbar": _make_array(3), "diboson": _make_array(2)}
    cfg = OmegaConf.create(
        {
            "samples": {
                "data": {"enabled": False},
                "signal": {"enabled": False},
            }
        }
    )
    grouped = group_samples(samples, cfg)

    assert len(grouped["data"]) == 0
    assert len(grouped["signal"]) == 0
    assert set(grouped["background"].keys()) == {"ttbar", "diboson"}


def test_split_mc_data():
    grouped = {
        "background": {"ttbar": _make_array(3)},
        "signal": {"GG_100": _make_array(2)},
        "data": {"data22": _make_array(4)},
    }
    mc, data = split_mc_data(grouped)

    assert set(mc.keys()) == {"ttbar", "GG_100"}
    assert set(data.keys()) == {"data22"}
