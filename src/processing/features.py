from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def resolve_features(
    cfg: DictConfig,
    *,
    fake: bool = False,
) -> list[str]:
    """Build a flat feature list from the Hydra features config.

    For the ML scope the config is structured by category (cleaning, truth,
    weights, tau, jet, kinematic, training, channel_1, channel_2, fake_weights).
    Channel-specific and fake-weight groups are appended based on the analysis
    channel and whether we are processing fake samples.

    For the NTuples / CC scopes the config contains a flat ``features`` list
    which is returned as-is.

    Parameters
    ----------
    cfg:
        Full Hydra config (must contain ``features`` and ``analysis`` groups).
    fake:
        Whether fake-weight features should be included.
    """
    feat_cfg = cfg.features
    scope = cfg.analysis.scope
    channel = str(cfg.analysis.channel) if cfg.analysis.channel is not None else None

    # NTuples and CC store a flat list under the ``features`` key.
    if scope in ("NTuples", "CC"):
        return list(OmegaConf.to_container(feat_cfg.features, resolve=True))

    # ML scope: assemble from category groups.
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

    if fake:
        features.extend(OmegaConf.to_container(feat_cfg.fake_weights, resolve=True))

    return features
