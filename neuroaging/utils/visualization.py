"""Visualization configuration and utilities for neuroaging figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from pathlib import Path

# Color palette for figures
COL_WEIGHTED = "#66c2a5"  # green - Set2[0]
COL_RAW = "#fc8d62"  # orange - Set2[1]
COL_REF = "#8da0cb"  # blue - Set2[2]

CMAP_WEIGHTED = "Greens"
CMAP_RAW = "Oranges"


def setup_plotting(
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 200,
    font_path: str | Path | None = None,
):
    """
    Configure matplotlib and seaborn for publication-quality figures.

    Parameters
    ----------
    figsize : tuple
        Default figure size in inches.
    dpi : int
        Display DPI (savefig uses 400 for print quality).
    font_path : str or Path, optional
        Path to custom font file (e.g., Calibri TTF).
    """
    # Add custom font if provided
    if font_path is not None:
        font_path = Path(font_path)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))

    mpl.rcParams.update(
        {
            # Canvas size & resolution
            "figure.figsize": figsize,
            "figure.dpi": dpi,
            "savefig.dpi": 400,
            # Fonts
            "font.family": "Calibri",
            "font.sans-serif": ["Calibri", "DejaVu Sans", "Arial"],
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 20,
            # Axis & spine aesthetics
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 1,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.8,
            # Colour cycle (colour-blind-safe)
            "axes.prop_cycle": mpl.cycler(color=sns.color_palette("Set2")),
            # Figure background
            "figure.facecolor": "white",
        }
    )

    # Seaborn theme inherits the rcParams above
    sns.set_theme(context="talk", style="whitegrid", palette="Set2")


def savefig_nice(
    fig: plt.Figure,
    filename: str | Path,
    *,
    tight: bool = True,
    transparent: bool = True,
    dpi: int = 300,
    **savefig_kwargs,
):
    """
    Save figure with consistent settings.

    Parameters
    ----------
    fig : matplotlib Figure
    filename : path to save to
    tight : apply tight_layout before saving
    transparent : transparent background
    dpi : output resolution
    **savefig_kwargs : passed to fig.savefig
    """
    if tight:
        fig.tight_layout()
    fig.savefig(filename, dpi=dpi, transparent=transparent, **savefig_kwargs)
    plt.close(fig)


# Metric display names
METRIC_LABELS = {
    "gm_vol": "GM Vol.",
    "wm_vol": "WM Vol.",
    "csf_vol": "CSF Vol.",
    "rd": "RD",
    "fa": "FA",
    "ad": "AD",
    "adc": "MD",
    "stacked": "Stacked",
}
