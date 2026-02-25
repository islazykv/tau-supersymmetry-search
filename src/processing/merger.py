from __future__ import annotations

import logging

import awkward as ak
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def merge_backgrounds(
    samples: dict[str, ak.Array],
    cfg: DictConfig,
) -> dict[str, ak.Array]:
    """Merge background samples according to the merge config.

    Parameters
    ----------
    samples:
        Dict of sample_id -> awkward array (unmerged backgrounds).
    cfg:
        Full Hydra config (needs ``merge.background_strategy`` and
        ``merge.primary_groups``).

    Returns
    -------
    Merged dict.  With ``as_one`` strategy a single ``'background'`` key is
    returned.  With ``as_primary`` the keys are the group names defined in the
    merge config.
    """
    strategy = cfg.merge.background_strategy

    if strategy == "as_one":
        merged = ak.concatenate(list(samples.values()), axis=0)
        return {"background": merged}

    if strategy == "as_primary":
        groups = OmegaConf.to_container(cfg.merge.primary_groups, resolve=True)
        out: dict[str, ak.Array] = {}
        for group_name, member_ids in groups.items():
            arrays = [samples[sid] for sid in member_ids if sid in samples]
            if arrays:
                out[group_name] = (
                    ak.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
                )
        return out

    raise ValueError(
        f"Unsupported background_strategy '{strategy}'. Use 'as_one' or 'as_primary'."
    )


def merge_fakes(samples: dict[str, ak.Array]) -> dict[str, ak.Array]:
    """Merge all fake samples into a single ``'faketaus'`` key."""
    merged = ak.concatenate(list(samples.values()), axis=0)
    return {"faketaus": merged}


def combine_background_fake(
    background: dict[str, ak.Array],
    fake: dict[str, ak.Array],
) -> dict[str, ak.Array]:
    """Add merged fakes to the background dict, dropping ff_weight columns."""
    fake_features_delete = {"ff_weight_1", "ff_weight_4"}
    faketaus = fake["faketaus"]
    faketaus = faketaus[[f for f in faketaus.fields if f not in fake_features_delete]]
    return {**background, "faketaus": faketaus}


def combine_background_signal(
    background: dict[str, ak.Array],
    signal: dict[str, ak.Array],
) -> dict[str, ak.Array]:
    """Combine background and signal dicts."""
    return {**background, **signal}


def assign_class(samples: dict[str, ak.Array]) -> None:
    """Add an integer ``'class'`` field to each sample (in-place)."""
    for i, key in enumerate(samples):
        samples[key]["class"] = i


def dict_to_array(samples: dict[str, ak.Array]) -> ak.Array:
    """Concatenate all samples into a single awkward array."""
    return ak.concatenate(list(samples.values()), axis=0)
