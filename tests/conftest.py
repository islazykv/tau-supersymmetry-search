from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.fixture()
def ml_cfg():
    """Minimal config with ML scope and channel 1 for feature/analysis tests."""
    return OmegaConf.create(
        {
            "analysis": {
                "run": "run2",
                "release": "r22",
                "analysis_base": "TauSUSY",
                "campaigns": ["mc20a"],
                "scope": "ML",
                "region": None,
                "channel": "1",
                "subject": None,
                "sub_subject": None,
            },
            "features": {
                "cleaning": ["eventClean", "isBadTile"],
                "truth": ["tau_isTruthMatchedTau"],
                "weights": ["mcEventWeight"],
                "training": ["nVtx"],
                "tau": ["tau_n", "tau_pt"],
                "jet": ["jet_n", "jet_pt"],
                "kinematic": ["met", "meff"],
                "channel_1": ["mu_n", "mu_pt"],
                "channel_2": ["Mt2", "sumMT"],
            },
            "samples": {
                "data": {"enabled": False, "samples": []},
                "background": {
                    "enabled": True,
                    "exclude": ["higgs"],
                    "samples": [
                        {"id": "ttbar", "label": "topquarks"},
                        {"id": "higgs", "label": "higgs"},
                        {"id": "diboson", "label": "diboson"},
                    ],
                },
                "signal": {"enabled": False},
            },
            "data": {"processed_path": "data/processed"},
            "merge": {
                "background_strategy": "as_one",
                "primary_groups": {
                    "topquarks": ["ttbar"],
                    "diboson": ["diboson"],
                },
            },
            "paths": {
                "background": "{analysis_base}/{campaign}/{sample}.root",
                "tree_name": "NOMINAL",
            },
        }
    )


def make_ak_array(n: int = 10, fields: list[str] | None = None) -> ak.Array:
    """Create a small awkward array with the given fields."""
    if fields is None:
        fields = ["met", "weight"]
    data = {f: np.random.default_rng(42).random(n) for f in fields}
    return ak.Array(data)
