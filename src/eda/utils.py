from __future__ import annotations

import pandas as pd

_DEFAULT_SIGNAL_TYPE_NAMES: dict[str, str] = {
    "GG": "gluinos",
    "SS": "squarks",
}


def get_class_names(
    df: pd.DataFrame,
    signal_type_names: dict[str, str] | None = None,
) -> list[str]:
    """Derive ordered class display names from eventOrigin prefixes, collapsing mixed-origin signal classes to a type label or 'signal'."""
    if signal_type_names is None:
        signal_type_names = _DEFAULT_SIGNAL_TYPE_NAMES

    class_names: list[str] = []
    for _cls, group in df.groupby("class", sort=True):
        origins = group["eventOrigin"].unique()

        if len(origins) == 1:
            class_names.append(origins[0])
            continue

        # Multiple origins → signal class; resolve from prefixes.
        prefixes = {o.split("_")[0] for o in origins}
        matched = {p: n for p, n in signal_type_names.items() if p in prefixes}

        if len(matched) == 1:
            class_names.append(next(iter(matched.values())))
        else:
            class_names.append("signal")

    return class_names
