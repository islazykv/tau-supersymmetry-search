import os
import shutil
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import atlas_mpl_style as ampl
import matplotlib.pyplot as plt


def apply_atlas_style():
    """Apply the ATLAS style and LaTeX compatibility."""

    cvmfs_path = "/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux"
    if os.path.exists(cvmfs_path):
        os.environ["PATH"] += ":" + cvmfs_path

    latex_exists = shutil.which("latex") is not None

    try:
        if latex_exists:
            plt.rc("text", usetex=True)
            ampl.use_atlas_style(usetex=True)
            print("ATLAS style applied with LaTeX.")
        else:
            plt.rc("text", usetex=False)
            ampl.use_atlas_style(usetex=False)
            print(
                "Notice: LaTeX not found on this system. Applying ATLAS style without LaTeX."
            )

    except Exception as e:
        print(f"Notice: ATLAS style fallback (Error: {e})")
        plt.rc("text", usetex=False)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
