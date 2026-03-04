import logging

import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

from src.processing.analysis import get_output_paths, resolve_samples  # noqa: E402
from src.processing.features import assign_event_origin  # noqa: E402
from src.processing.io import save_samples  # noqa: E402
from src.processing.merger import merge_backgrounds, merge_signals  # noqa: E402
from src.processing.processor import process_samples  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Run the full preprocessing pipeline: process, merge, assign event origin, and save."""
    sample_lists = resolve_samples(cfg)

    log.info(
        "Resolved samples — background: %d, signal: %d",
        len(sample_lists["background"]),
        len(sample_lists["signal"]),
    )

    # --- process ---
    data = process_samples(cfg, "data", [s.id for s in sample_lists["data"]])
    log.info("Processed %d data sample(s)", len(data))

    background = process_samples(
        cfg, "background", [s.id for s in sample_lists["background"]]
    )
    log.info("Processed %d background samples", len(background))

    signal = process_samples(cfg, "signal", [s.id for s in sample_lists["signal"]])
    log.info("Processed %d signal samples", len(signal))

    # --- group ---
    samples = {"data": data, "background": background, "signal": signal}

    # --- merge backgrounds ---
    samples["background"] = merge_backgrounds(samples["background"], cfg)
    log.info("Merged backgrounds into %d group(s)", len(samples["background"]))

    # --- event origin ---
    assign_event_origin(samples)

    # --- merge signals ---
    samples["signal"] = merge_signals(samples["signal"], cfg)
    log.info("Merged signal into %d group(s)", len(samples["signal"]))

    # --- save ---
    output_paths = get_output_paths(cfg)
    samples_dir = output_paths["samples_dir"]
    save_samples(samples["data"], samples_dir)
    save_samples(samples["background"], samples_dir)
    save_samples(samples["signal"], samples_dir)
    log.info("Preprocessing complete — output saved to %s", samples_dir)


if __name__ == "__main__":
    main()
