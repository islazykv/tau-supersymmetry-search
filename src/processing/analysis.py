from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig


@dataclass(frozen=True)
class Sample:
    id: str
    label: str


@dataclass(frozen=True)
class AnalysisConfig:
    run: str
    release: str
    analysis_base: str
    campaigns: list[str]
    scope: str
    region: str | None = None
    channel: str | None = None
    subject: str | None = None
    sub_subject: str | None = None


def _apply_excludes(samples: list[dict], excludes: list[str]) -> list[Sample]:
    """Filter sample dicts and return Sample objects, removing excluded IDs."""
    return [
        Sample(id=s["id"], label=s["label"]) for s in samples if s["id"] not in excludes
    ]


def _discover_signals(scan_path: str, filter_patterns: list[str]) -> list[Sample]:
    """Discover signal samples from disk, matching any of the filter patterns."""
    if not os.path.isdir(scan_path):
        return []

    signals: list[Sample] = []
    for entry in sorted(os.listdir(scan_path)):
        if any(pat in entry for pat in filter_patterns):
            signal_name = entry.split(".")[0]
            signal_label = "_".join(signal_name.split("_")[0:3])
            signals.append(Sample(id=signal_name, label=signal_label))
    return signals


def resolve_samples(cfg: DictConfig) -> dict[str, list[Sample]]:
    """Build categorized sample lists from Hydra config."""
    result: dict[str, list[Sample]] = {}

    data_cfg = cfg.samples.data
    if data_cfg.get("enabled", False):
        result["data"] = [
            Sample(id=s["id"], label=s["label"]) for s in data_cfg.samples
        ]
    else:
        result["data"] = []

    bg_cfg = cfg.samples.background
    if bg_cfg.get("enabled", False):
        excludes = list(bg_cfg.get("exclude", []))
        result["background"] = _apply_excludes(bg_cfg.samples, excludes)
    else:
        result["background"] = []

    sig_cfg = cfg.samples.signal
    if sig_cfg.get("enabled", False):
        campaign = cfg.analysis.campaigns[0]
        scan_path = sig_cfg.scan_path.format(
            analysis_base=cfg.analysis.analysis_base,
            campaign=campaign,
        )
        filter_patterns = list(sig_cfg.filter_patterns)
        result["signal"] = _discover_signals(scan_path, filter_patterns)
    else:
        result["signal"] = []

    return result


def get_output_paths(cfg: DictConfig) -> dict[str, Path]:
    """Compute output directory paths from analysis config."""
    analysis = cfg.analysis
    base = (
        Path(cfg.data.processed_path)
        / analysis.scope
        / analysis.analysis_base
        / analysis.run
        / (analysis.region or "")
        / (f"{analysis.channel}_tau" if analysis.channel else "")
    )
    return {
        "samples_dir": base / "samples",
        "dataframes_dir": base / "dataframes",
        "plots_dir": base / "plots",
        "models_dir": base / "models",
    }
