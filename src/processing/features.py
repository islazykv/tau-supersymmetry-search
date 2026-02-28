from __future__ import annotations

import awkward as ak
from omegaconf import DictConfig, OmegaConf


def extract_feature_from_array(array_in: ak.Array, feature_name: str) -> ak.Array:
    """Extract a single feature field from an awkward array."""
    return array_in[feature_name]


def drop_features(array_in: ak.Array, feature_list: list[str]) -> ak.Array:
    """Return the array with the specified feature fields removed."""
    return array_in[[f for f in ak.fields(array_in) if f not in feature_list]]


def assign_event_origin(grouped: dict[str, dict[str, ak.Array]]) -> None:
    """Add an ``'eventOrigin'`` field to each array in-place, set to the sample id."""
    for category in grouped.values():
        for sid, array in category.items():
            category[sid] = ak.with_field(array, sid, "eventOrigin")


def resolve_features(cfg: DictConfig) -> list[str]:
    """Build and return a flat feature list from the Hydra features config based on scope and channel."""
    feat_cfg = cfg.features
    scope = cfg.analysis.scope
    channel = str(cfg.analysis.channel) if cfg.analysis.channel is not None else None

    if scope in ("NTuples", "CC"):
        return list(OmegaConf.to_container(feat_cfg.features, resolve=True))

    features: list[str] = []
    for group in (
        "cleaning",
        "truth",
        "weights",
        "training",
        "tau",
        "jet",
        "kinematic",
    ):
        features.extend(OmegaConf.to_container(feat_cfg[group], resolve=True))

    if channel in ("1had1lep", "1had0lep", "1"):
        features.extend(OmegaConf.to_container(feat_cfg.channel_1, resolve=True))
    elif channel == "2":
        features.extend(OmegaConf.to_container(feat_cfg.channel_2, resolve=True))

    return features
