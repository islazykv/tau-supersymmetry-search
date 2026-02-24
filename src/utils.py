import warnings
import pandas as pd
import awkward as ak
from pathlib import Path


def suppress_warnings():
    """Suppress unessential warnings."""
    warnings.filterwarnings("ignore")
    print("Unessential warnings suppressed.")


def save_data(data, output_path: str):
    """
    Saves data to Parquet format.
    Handles both Awkward Arrays and Pandas DataFrames.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_parquet(path)
    elif isinstance(data, ak.Array):
        ak.to_parquet(data, path)

    print(f"File saved to: {output_path}")


def load_data(file_path: str, is_awkward: bool = False):
    """Loads data from Parquet format."""
    if is_awkward:
        return ak.from_parquet(file_path)
    return pd.read_parquet(file_path)
