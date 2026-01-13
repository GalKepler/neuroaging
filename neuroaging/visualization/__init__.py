"""
Visualization utilities for neuroaging project.

This package provides consistent plotting configuration, color schemes, and
visualization functions used across all analysis notebooks.
"""

from .config import (
    COL_CENSUS,
    COL_RAW,
    COL_REF,
    COL_WEIGHTED,
    CMAP_RAW,
    CMAP_WEIGHTED,
    configure_plotting,
    savefig_nice,
)
from .demographics import get_dist, plot_representative_pyramid

__all__ = [
    "configure_plotting",
    "savefig_nice",
    "COL_RAW",
    "COL_WEIGHTED",
    "COL_REF",
    "COL_CENSUS",
    "CMAP_RAW",
    "CMAP_WEIGHTED",
    "get_dist",
    "plot_representative_pyramid",
]
