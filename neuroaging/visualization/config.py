"""
Visualization configuration and utilities for neuroaging project.

This module provides consistent matplotlib/seaborn configuration across all notebooks
and figures, eliminating ~100 lines of duplicated setup code in 24+ notebooks.
"""

import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager
import numpy as np

# Export color constants
COL_RAW = "#ffb300"
COL_WEIGHTED = "#7F0099CE"
COL_REF = "0.25"
COL_CENSUS = "#A5A5A5"
COL_WHITE = "white"

# Generate colormaps
COL_RAW_RGB = mpl.colors.to_rgb(COL_RAW)
COL_WEIGHTED_RGB = mpl.colors.to_rgb(COL_WEIGHTED)
CMAP_RAW = mpl.colors.LinearSegmentedColormap.from_list("raw", [COL_WHITE, COL_RAW_RGB], N=256)
CMAP_WEIGHTED = mpl.colors.LinearSegmentedColormap.from_list(
    "weighted", [COL_WHITE, COL_WEIGHTED_RGB], N=256
)


def configure_plotting(
    font_family="Calibri",
    font_path=None,
    figure_size=(12, 8),
    figure_dpi=200,
    savefig_dpi=400,
    style="whitegrid",
    palette="Set2",
):
    """
    Configure matplotlib and seaborn for consistent plotting across notebooks.

    Call this once at the start of your notebook to set up all plotting parameters.

    Parameters
    ----------
    font_family : str, default="Calibri"
        Font family to use for all text
    font_path : str or None, optional
        Path to custom font file to register
    figure_size : tuple, default=(12, 8)
        Default figure size in inches (width, height)
    figure_dpi : int, default=200
        DPI for figure display
    savefig_dpi : int, default=400
        DPI for saved figures
    style : str, default="whitegrid"
        Seaborn style name
    palette : str, default="Set2"
        Seaborn color palette name

    Examples
    --------
    >>> from neuroaging.visualization import configure_plotting
    >>> configure_plotting()
    >>> # Now all matplotlib/seaborn plots will use consistent styling
    """
    # Register custom font if provided
    if font_path:
        font_manager.fontManager.addfont(font_path)

    mpl.rcParams.update(
        {
            "figure.figsize": figure_size,
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
            "font.family": font_family,
            "font.sans-serif": [font_family, "DejaVu Sans", "Arial"],
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 20,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 1,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.8,
            "axes.prop_cycle": mpl.cycler(color=sns.color_palette(palette)),
            "figure.facecolor": "white",
        }
    )

    sns.set_theme(context="talk", style=style, palette=palette)


def savefig_nice(fig, filename, *, tight=True, transparent=True, dpi=300, **kwargs):
    """
    Save figure with consistent formatting.

    This is a convenience wrapper around fig.savefig() that applies consistent
    default parameters used across all project figures.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filename : str or Path
        Output file path
    tight : bool, default=True
        Whether to apply tight_layout before saving
    transparent : bool, default=True
        Whether to save with transparent background
    dpi : int, default=300
        Resolution in dots per inch
    **kwargs
        Additional keyword arguments passed to fig.savefig()

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from neuroaging.visualization import savefig_nice
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> savefig_nice(fig, "output/figure1.png")
    """
    if tight:
        fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", transparent=transparent, **kwargs)


__all__ = [
    "configure_plotting",
    "savefig_nice",
    "COL_RAW",
    "COL_WEIGHTED",
    "COL_REF",
    "COL_CENSUS",
    "CMAP_RAW",
    "CMAP_WEIGHTED",
]
