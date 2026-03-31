from __future__ import annotations

import pandas as pd

_DEFAULT_SIGNAL_TYPE_NAMES: dict[str, str] = {
    "GG": "gluinos",
    "SS": "squarks",
}


def get_class_names(df: pd.DataFrame) -> list[str]:
    """Derive ordered internal class names from eventOrigin."""
    class_names: list[str] = []
    for _cls, group in df.groupby("class", sort=True):
        origins = group["eventOrigin"].unique()
        class_names.append(origins[0] if len(origins) == 1 else "signal")
    return class_names


def get_class_labels(
    df: pd.DataFrame,
    display_labels: dict[str, str] | None = None,
    signal_type_names: dict[str, str] | None = None,
) -> list[str]:
    """Derive ordered display labels from eventOrigin."""
    if signal_type_names is None:
        signal_type_names = _DEFAULT_SIGNAL_TYPE_NAMES
    if display_labels is None:
        display_labels = {}

    class_labels: list[str] = []
    for _cls, group in df.groupby("class", sort=True):
        origins = group["eventOrigin"].unique()

        if len(origins) == 1:
            origin = origins[0]
            class_labels.append(display_labels.get(origin, origin))
            continue

        # Multiple origins → signal class; resolve from prefixes
        prefixes = {o.split("_")[0] for o in origins}
        matched = {p: n for p, n in signal_type_names.items() if p in prefixes}

        if len(matched) == 1:
            class_labels.append(next(iter(matched.values())))
        else:
            class_labels.append("signal")

    return class_labels
