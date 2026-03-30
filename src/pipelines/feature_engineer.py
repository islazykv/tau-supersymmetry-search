"""Feature engineering pipeline: load, split, rectify, and save."""

from __future__ import annotations

import logging

import pyrootutils
from omegaconf import DictConfig

from src.processing.analysis import get_output_paths
from src.processing.features import (
    assign_class_weights,
    drop_features,
    extract_feature_from_array,
    resolve_features_to_drop,
)
from src.processing.io import load_samples, save_dataframe
from src.processing.merger import (
    assign_class,
    dict_to_array,
    group_samples,
    split_mc_data,
)
from src.processing.rectangularizer import fill_padding, rectangularize_pad_array
from src.processing.validation import validate_mc

log = logging.getLogger(__name__)


def feature_engineer(cfg: DictConfig) -> None:
    """Run the full feature engineering pipeline: load, split, rectify, and save."""
    root = pyrootutils.find_root(indicator=[".git", "pyproject.toml"])
    output_paths = get_output_paths(cfg)
    input_dir = root / output_paths["samples_dir"]

    sample_ids = [f.stem for f in input_dir.glob("*.parquet")]
    samples = load_samples(input_dir, sample_ids)
    log.info("Loaded %d samples", len(samples))

    grouped = group_samples(samples, cfg)
    log.info(
        "Grouped samples — data: %d, background: %d, signal: %d",
        len(grouped["data"]),
        len(grouped["background"]),
        len(grouped["signal"]),
    )

    samples_mc, samples_data = split_mc_data(grouped)
    log.info(
        "Split into MC (%d sample(s)) and data (%d sample(s))",
        len(samples_mc),
        len(samples_data),
    )

    assign_class(samples_mc)

    samples_mc = dict_to_array(samples_mc)
    samples_data = dict_to_array(samples_data)

    event_origin = extract_feature_from_array(samples_mc, "eventOrigin")
    tau_n = extract_feature_from_array(samples_mc, "tau_n")

    features_to_drop = resolve_features_to_drop(cfg)
    samples_mc = drop_features(samples_mc, features_to_drop)
    samples_data = drop_features(samples_data, features_to_drop)
    log.info("Dropped %d features", len(features_to_drop))

    df_mc = rectangularize_pad_array(
        array_in=samples_mc,
        padding_threshold=cfg.data.padding_threshold,
        nan_threshold=cfg.data.nan_threshold,
    )
    df_data = rectangularize_pad_array(
        array_in=samples_data,
        padding_threshold=cfg.data.padding_threshold,
        nan_threshold=0.0,
    )
    log.info("Rectangularized MC: %d rows, %d columns", len(df_mc), len(df_mc.columns))
    log.info(
        "Rectangularized data: %d rows, %d columns",
        len(df_data),
        len(df_data.columns),
    )

    df_data = df_data[df_mc.columns.intersection(df_data.columns)]

    df_mc = fill_padding(df=df_mc, strategy="0")
    df_data = fill_padding(df=df_data, strategy="0")

    df_mc["eventOrigin"] = event_origin
    df_mc["tau_n"] = tau_n

    weights = assign_class_weights(df_mc)
    log.info("Class weights: %s", weights.to_dict())

    df_mc = validate_mc(df_mc)

    dataframes_dir = root / output_paths["dataframes_dir"]
    save_dataframe(df=df_mc, path=dataframes_dir / "mc.parquet")
    save_dataframe(df=df_data, path=dataframes_dir / "data.parquet")
    log.info("Feature engineering complete — output saved to %s", dataframes_dir)
