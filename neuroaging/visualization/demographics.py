"""
Demographics visualization utilities for creating population pyramids.

This module provides functions for visualizing age and sex distributions
comparing sample data to census/population data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple


def get_dist(
    df: pd.DataFrame,
    census_df: pd.DataFrame,
    weight_col: str = None,
    age_col: str = "age_at_scan",
    sex_col: str = "sex",
) -> Tuple[pd.Series, pd.Series]:
    """
    Get age distribution by sex from sample data.

    Parameters
    ----------
    df : pd.DataFrame
        Sample data with age and sex columns
    census_df : pd.DataFrame
        Census data with age bins (range_start, range_end) for bin definition
    weight_col : str, optional
        Column name for weights. If None, uses unweighted counts
    age_col : str, default="age_at_scan"
        Column name for age
    sex_col : str, default="sex"
        Column name for sex (assumes Male=0, Female=1)

    Returns
    -------
    male_dist : pd.Series
        Male distribution as percentage, indexed by age bin start
    female_dist : pd.Series
        Female distribution as percentage, indexed by age bin start
    """
    # Get bin edges from census data
    labels = sorted(census_df["range_start"].unique())
    edges = sorted(labels + [int(census_df["range_end"].max()) + 1])

    df = df.copy()
    df["bin"] = pd.cut(df[age_col], bins=edges, labels=labels, right=False)

    if weight_col:
        counts = df.groupby(["bin", sex_col], observed=False)[weight_col].sum().reset_index()
    else:
        counts = df.groupby(["bin", sex_col], observed=False).size().reset_index(name="c")

    total = counts.iloc[:, 2].sum()
    male_dist = counts[counts[sex_col] == 0].set_index("bin").iloc[:, 1] / total * 100
    female_dist = counts[counts[sex_col] == 1].set_index("bin").iloc[:, 1] / total * 100

    return male_dist, female_dist


def plot_representative_pyramid(
    snbb_df: pd.DataFrame,
    census_df: pd.DataFrame,
    age_col: str = "age_at_scan",
    sex_col: str = "sex",
    weight_col: str = "weight",
    figsize: Tuple[int, int] = (18, 6),
    title: str = "(B) Sex distribution across the lifespan",
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create a population pyramid comparing census, raw sample, and weighted sample.

    Parameters
    ----------
    snbb_df : pd.DataFrame
        Sample data with age, sex, and weight columns
    census_df : pd.DataFrame
        Census data with columns: range_start, range_end, male, female
    age_col : str, default="age_at_scan"
        Column name for age in sample data
    sex_col : str, default="sex"
        Column name for sex in sample data (Male=0, Female=1)
    weight_col : str, default="weight"
        Column name for post-stratification weights
    figsize : tuple, default=(18, 6)
        Figure size (width, height) in inches
    title : str, optional
        Suptitle for the figure

    Returns
    -------
    fig : plt.Figure
        Main figure with three population pyramids
    fig_legend : plt.Figure
        Separate legend figure

    Notes
    -----
    Creates three side-by-side population pyramids:
    - (I) Census/population reference distribution
    - (II) Raw sample (unweighted) distribution
    - (III) Weighted sample distribution

    Examples
    --------
    >>> fig, fig_legend = plot_representative_pyramid(sample_df, census_df)
    >>> fig.savefig("population_pyramid.png", dpi=300)
    >>> fig_legend.savefig("pyramid_legend.png", dpi=300)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    # 1. Prepare Census Data
    pop_total = census_df["male"].sum() + census_df["female"].sum()
    census_m = (
        census_df.sort_values("range_start").set_index("range_start")["male"] / pop_total
    ) * 100
    census_f = (
        census_df.sort_values("range_start").set_index("range_start")["female"] / pop_total
    ) * 100
    labels = census_m.index.tolist()

    # 2. Get distributions from sample
    raw_m, raw_f = get_dist(snbb_df, census_df, weight_col=None, age_col=age_col, sex_col=sex_col)
    weighted_m, weighted_f = get_dist(
        snbb_df, census_df, weight_col=weight_col, age_col=age_col, sex_col=sex_col
    )

    titles = ["(I) Israel census (ref)", "(II) SNBB sample (unweighted)", "(III) SNBB sample (weighted)"]
    data = [(census_m, census_f), (raw_m, raw_f), (weighted_m, weighted_f)]

    for i, ax in enumerate(axes):
        m, f = data[i]
        y = np.arange(len(labels))
        ax.barh(y, -m, color="skyblue", label="Male", edgecolor="black", alpha=0.8)
        ax.barh(y, f, color="salmon", label="Female", edgecolor="black", alpha=0.8)
        ax.set_title(titles[i], fontsize=30, pad=20)
        ax.set_xlabel("Percentage (%)", fontsize=24)
        ax.axvline(0, color="black", linewidth=1)

        if i == 0:
            ax.set_ylabel("Age Bin (Start Year)", fontsize=24)
            # Show every other label for clarity
            ax.set_yticks(y[::2])
            ax.set_yticklabels(labels[::2])

        # Symmetrize X-axis
        limit = max(m.max(), f.max()) + 1
        ax.set_xlim(-limit, limit)

    plt.suptitle(title, fontsize=36, y=0.95)
    plt.tight_layout()

    # Create separate legend
    fig_legend = plt.figure(figsize=(4, 2))
    handles = [
        plt.Line2D([0], [0], color="skyblue", lw=10, label="Male"),
        plt.Line2D([0], [0], color="salmon", lw=10, label="Female"),
    ]
    fig_legend.legend(handles=handles, loc="center", fontsize=20)
    fig_legend.tight_layout()

    return fig, fig_legend


__all__ = ["get_dist", "plot_representative_pyramid"]
