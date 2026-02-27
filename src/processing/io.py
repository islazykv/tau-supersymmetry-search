from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak

log = logging.getLogger(__name__)


def save_samples(samples: dict[str, ak.Array], output_dir: Path) -> None:
    """Save each sample array as a parquet file under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id, array in samples.items():
        path = output_dir / f"{sample_id}.parquet"
        ak.to_parquet(array, str(path))
        log.info("Saved %s (%d events)", path, len(array))


def load_samples(
    input_dir: Path,
    sample_ids: list[str],
) -> dict[str, ak.Array]:
    """Load parquet files from input_dir and return a dict keyed by sample id."""
    input_dir = Path(input_dir)
    out: dict[str, ak.Array] = {}

    for sid in sample_ids:
        path = input_dir / f"{sid}.parquet"
        out[sid] = ak.from_parquet(str(path))
        log.info("Loaded %s (%d events)", path, len(out[sid]))

    return out
