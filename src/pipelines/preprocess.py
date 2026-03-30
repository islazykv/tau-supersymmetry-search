"""Preprocessing pipeline: ROOT → cuts → weights → merge → save."""

from __future__ import annotations

import logging

import pyrootutils
from omegaconf import DictConfig

from src.processing.analysis import get_output_paths, resolve_samples
from src.processing.features import assign_event_origin
from src.processing.io import save_samples
from src.processing.merger import merge_backgrounds, merge_signals
from src.processing.processor import process_samples

log = logging.getLogger(__name__)


def preprocess(cfg: DictConfig) -> None:
    """Run the full preprocessing pipeline: process, merge, assign event origin, and save."""
    sample_lists = resolve_samples(cfg)

    log.info(
        "Resolved samples — background: %d, signal: %d",
        len(sample_lists["background"]),
        len(sample_lists["signal"]),
    )

    data = process_samples(cfg, "data", [s.id for s in sample_lists["data"]])
    log.info("Processed %d data sample(s)", len(data))

    background = process_samples(
        cfg, "background", [s.id for s in sample_lists["background"]]
    )
    log.info("Processed %d background samples", len(background))

    signal = process_samples(cfg, "signal", [s.id for s in sample_lists["signal"]])
    log.info("Processed %d signal samples", len(signal))

    samples = {"data": data, "background": background, "signal": signal}

    samples["background"] = merge_backgrounds(samples["background"], cfg)
    log.info("Merged backgrounds into %d group(s)", len(samples["background"]))

    assign_event_origin(samples)

    samples["signal"] = merge_signals(samples["signal"], cfg)
    log.info("Merged signal into %d group(s)", len(samples["signal"]))

    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    samples_dir = root / output_paths["samples_dir"]
    save_samples(samples["data"], samples_dir)
    save_samples(samples["background"], samples_dir)
    save_samples(samples["signal"], samples_dir)
    log.info("Preprocessing complete — output saved to %s", samples_dir)
