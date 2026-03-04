import logging

import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    cwd=True,
)

from src.processing.analysis import get_output_paths  # noqa: E402
from src.processing.features import (  # noqa: E402
    assign_class_weights,
    drop_features,
    extract_feature_from_array,
    resolve_features_to_drop,
)
from src.processing.io import load_samples, save_dataframe  # noqa: E402
from src.processing.merger import (  # noqa: E402
    assign_class,
    dict_to_array,
    group_samples,
    split_mc_data,
)
from src.processing.rectangularizer import fill_padding, rectangularize_pad_array  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Run the full feature engineering pipeline: load, split, rectify, and save."""
    output_paths = get_output_paths(cfg)
    input_dir = output_paths["samples_dir"]

    # --- load ---
    sample_ids = [f.stem for f in input_dir.glob("*.parquet")]
    samples = load_samples(input_dir, sample_ids)
    log.info("Loaded %d samples", len(samples))

    # --- group ---
    grouped = group_samples(samples, cfg)
    log.info(
        "Grouped samples — data: %d, background: %d, signal: %d",
        len(grouped["data"]),
        len(grouped["background"]),
        len(grouped["signal"]),
    )

    # --- split ---
    samples_mc, samples_data = split_mc_data(grouped)
    log.info(
        "Split into MC (%d sample(s)) and data (%d sample(s))",
        len(samples_mc),
        len(samples_data),
    )

    # --- class ---
    assign_class(samples_mc)

    # --- flatten ---
    samples_mc = dict_to_array(samples_mc)
    samples_data = dict_to_array(samples_data)

    # --- extract ---
    event_origin = extract_feature_from_array(samples_mc, "eventOrigin")
    tau_n = extract_feature_from_array(samples_mc, "tau_n")

    # --- drop ---
    features_to_drop = resolve_features_to_drop(cfg)
    samples_mc = drop_features(samples_mc, features_to_drop)
    samples_data = drop_features(samples_data, features_to_drop)
    log.info("Dropped %d features", len(features_to_drop))

    # --- rectangularize ---
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
    log.info("Rectangularized data: %d rows, %d columns", len(df_data), len(df_data.columns))

    # --- align ---
    df_data = df_data[df_mc.columns.intersection(df_data.columns)]

    # --- fill ---
    df_mc = fill_padding(df=df_mc, strategy="0")
    df_data = fill_padding(df=df_data, strategy="0")

    # --- reattach ---
    df_mc["eventOrigin"] = event_origin
    df_mc["tau_n"] = tau_n

    # --- class weights ---
    weights = assign_class_weights(df_mc)
    log.info("Class weights: %s", weights.to_dict())

    # --- save ---
    dataframes_dir = output_paths["dataframes_dir"]
    save_dataframe(df=df_mc, path=dataframes_dir / "mc.parquet")
    save_dataframe(df=df_data, path=dataframes_dir / "data.parquet")
    log.info("Feature engineering complete — output saved to %s", dataframes_dir)


if __name__ == "__main__":
    main()
