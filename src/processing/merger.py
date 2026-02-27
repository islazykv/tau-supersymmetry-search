from __future__ import annotations

import logging

import awkward as ak
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def merge_backgrounds(
    samples: dict[str, ak.Array],
    cfg: DictConfig,
) -> dict[str, ak.Array]:
    """Merge background samples into groups according to the strategy defined in cfg."""
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
