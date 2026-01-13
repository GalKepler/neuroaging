#!/usr/bin/env python
"""
Example showing BEFORE and AFTER for fig1.ipynb updates.

This script shows exactly what code to find and replace in the notebooks.
"""

print("=" * 80)
print("FIG1.IPYNB UPDATE EXAMPLE")
print("=" * 80)
print()

print("📋 STEP 1: Find and Replace Matplotlib Configuration")
print("=" * 80)
print()
print("❌ BEFORE (DELETE ~100 lines):")
print("-" * 80)
print('''
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# Register fonts
font_path = "/home/galkepler/.fonts/calibri.ttf"
font_manager.fontManager.addfont(font_path)

# Color constants
COL_RAW = "#ffb300"
COL_WEIGHTED = "#7F0099CE"
COL_REF = "0.25"
COL_CENSUS = "#A5A5A5"

# Colormaps
COL_RAW_RGB = mpl.colors.to_rgb(COL_RAW)
COL_WEIGHTED_RGB = mpl.colors.to_rgb(COL_WEIGHTED)
CMAP_RAW = mpl.colors.LinearSegmentedColormap.from_list("raw", ["white", COL_RAW_RGB], N=256)
CMAP_WEIGHTED = mpl.colors.LinearSegmentedColormap.from_list("weighted", ["white", COL_WEIGHTED_RGB], N=256)

# Configure matplotlib
mpl.rcParams.update({
    "figure.figsize": (12, 8),
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "font.family": "Calibri",
    "font.sans-serif": ["Calibri", "DejaVu Sans", "Arial"],
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.4,
    "axes.prop_cycle": mpl.cycler(color=sns.color_palette("Set2")),
    "figure.facecolor": "white",
    # ... more lines
})

sns.set_theme(context="talk", style="whitegrid", palette="Set2")

def savefig_nice(fig, filename, *, tight=True, transparent=True, dpi=300, **kwargs):
    """Save figure with consistent formatting."""
    if tight:
        fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", transparent=transparent, **kwargs)
''')

print()
print("✅ AFTER (11 lines):")
print("-" * 80)
print('''
import matplotlib.pyplot as plt
from neuroaging.visualization import (
    configure_plotting,
    savefig_nice,
    COL_RAW,
    COL_WEIGHTED,
    COL_REF,
    COL_CENSUS,
    CMAP_RAW,
    CMAP_WEIGHTED,
)

configure_plotting()
''')

print()
print("💾 Savings: ~89 lines eliminated!")
print()

print("=" * 80)
print("📋 STEP 2: Find and Replace compute_poststrat_weights")
print("=" * 80)
print()
print("❌ BEFORE (DELETE ~130 lines):")
print("-" * 80)
print('''
def compute_poststrat_weights(
    sample_counts: pd.DataFrame,
    population_counts: pd.DataFrame,
    strata: Union[str, List[str]],
    return_summary: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
    """
    Compute post-stratification weights to match population distribution.
    
    Parameters
    ----------
    sample_counts : pd.DataFrame
        Sample data with strata columns and counts
    population_counts : pd.DataFrame  
        Population data with strata columns and counts
    strata : str or list of str
        Column name(s) defining strata (e.g., ["age_group", "sex"])
    return_summary : bool
        Whether to return summary DataFrame
        
    Returns
    -------
    weights : pd.Series
        Post-stratification weight for each stratum
    """
    # Validate inputs
    if isinstance(strata, str):
        strata = [strata]
        
    # ... 120+ more lines of implementation ...
    
    return weights
''')

print()
print("✅ AFTER (1 line):")
print("-" * 80)
print('''
from neuroaging.stats import compute_poststrat_weights
''')

print()
print("💾 Savings: ~130 lines eliminated!")
print()

print("=" * 80)
print("📋 COMPLETE IMPORT SECTION (after changes)")
print("=" * 80)
print()
print("Your import cell should look like this:")
print("-" * 80)
print('''
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Refactored visualization imports
from neuroaging.visualization import (
    configure_plotting,
    savefig_nice,
    COL_RAW,
    COL_WEIGHTED,
    COL_REF,
    COL_CENSUS,
    CMAP_RAW,
    CMAP_WEIGHTED,
)

# Refactored stats imports
from neuroaging.stats import compute_poststrat_weights

# Configure plotting
configure_plotting()

# Your other imports...
''')

print()
print("=" * 80)
print("✅ SUMMARY FOR FIG1.IPYNB")
print("=" * 80)
print()
print("Changes:")
print("  • Replace matplotlib config block (100 lines → 11 lines)")
print("  • Replace compute_poststrat_weights definition (130 lines → 1 line)")
print()
print("Total savings: ~219 lines")
print()
print("Next steps:")
print("  1. Make a backup: cp fig1.ipynb fig1_backup.ipynb")
print("  2. Open fig1.ipynb in JupyterLab")
print("  3. Find and replace the two blocks shown above")
print("  4. Run 'Restart Kernel and Run All Cells'")
print("  5. Verify figures look the same")
print()
print("=" * 80)

