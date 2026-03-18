from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.processing.features import assign_class_weights, resolve_features


def test_ml_channel1_includes_channel1_features(ml_cfg):
    features = resolve_features(ml_cfg)
    assert "mu_n" in features
    assert "mu_pt" in features
    assert "Mt2" not in features
    assert "sumMT" not in features


def test_ml_channel2_includes_channel2_features(ml_cfg):
    OmegaConf.update(ml_cfg, "analysis.channel", "2")
    features = resolve_features(ml_cfg)
    assert "Mt2" in features
    assert "sumMT" in features
    assert "mu_n" not in features


def test_ml_includes_base_groups(ml_cfg):
    features = resolve_features(ml_cfg)
    for f in [
        "eventClean",
        "isBadTile",
        "tau_isTruthMatchedTau",
        "mcEventWeight",
        "nVtx",
        "tau_n",
        "jet_n",
        "met",
    ]:
        assert f in features


def test_ntuples_scope_returns_flat_list():
    cfg = OmegaConf.create(
        {
            "analysis": {"scope": "NTuples", "channel": None},
            "features": {"features": ["feat_a", "feat_b", "feat_c"]},
        }
    )
    features = resolve_features(cfg)
    assert features == ["feat_a", "feat_b", "feat_c"]


def test_assign_class_weights_balanced():
    df = pd.DataFrame({"class": [0, 0, 1, 1]})
    weights = assign_class_weights(df)

    assert (weights == 1.0).all()
    assert (df["class_weight"] == 1.0).all()


def test_assign_class_weights_imbalanced():
    df = pd.DataFrame({"class": [0, 0, 0, 1]})
    weights = assign_class_weights(df)

    assert weights[1] == pytest.approx(1.0)
    assert weights[0] == pytest.approx(1 / 3)
    assert df.loc[df["class"] == 0, "class_weight"].iloc[0] == pytest.approx(1 / 3)
    assert df.loc[df["class"] == 1, "class_weight"].iloc[0] == pytest.approx(1.0)


def test_assign_class_weights_missing_column():
    df = pd.DataFrame({"not_class": [1, 2]})

    with pytest.raises(ValueError, match="must contain a 'class' column"):
        assign_class_weights(df)
