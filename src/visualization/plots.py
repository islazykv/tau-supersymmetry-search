import logging
import os
import shutil
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import atlas_mpl_style as ampl
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def apply_atlas_style():
    """Apply the ATLAS matplotlib style with LaTeX if available."""

    cvmfs_path = "/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux"
    if os.path.exists(cvmfs_path):
        os.environ["PATH"] += ":" + cvmfs_path

    latex_exists = shutil.which("latex") is not None

    try:
        if latex_exists:
            plt.rc("text", usetex=True)
            ampl.use_atlas_style(usetex=True)
            log.info("ATLAS style applied with LaTeX.")
        else:
            plt.rc("text", usetex=False)
            ampl.use_atlas_style(usetex=False)
            log.info("LaTeX not found — applying ATLAS style without LaTeX.")

    except Exception as e:
        log.warning("ATLAS style fallback (Error: %s)", e)
        plt.rc("text", usetex=False)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure as PNG and PDF, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )


def save_plotly_figure(
    fig, path: Path, width: int = 1200, height: int = 700, scale: int = 5
) -> None:
    """Save a Plotly figure as PNG and show it as a static image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(width=width, height=height)
    fig.write_image(path, scale=scale)
    fig.show(renderer="png", width=width, height=height)
