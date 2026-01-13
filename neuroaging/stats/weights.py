"""
Post-stratification weighting utilities.

This module provides functions for computing post-stratification weights to adjust
sample demographics to match population distributions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Re-export the existing function from utils
from neuroaging.utils.utils import compute_poststrat_weights


def assign_weights_by_population(
    data: pd.DataFrame,
    population_df: pd.DataFrame,
    *,
    age_col: str = "age_at_scan",
    range_start: str = "range_start",
    range_end: str = "range_end",
    total: str = "total",
) -> np.ndarray:
    """
    Post-stratification weights so the age distribution of *data*
    matches an external reference.

    Parameters
    ----------
    data : pd.DataFrame
        Sample data with age column
    population_df : pd.DataFrame
        Population data with age bins [range_start, range_end] and total column
    age_col : str, default="age_at_scan"
        Column name for age in data
    range_start : str, default="range_start"
        Column name for bin start in population_df
    range_end : str, default="range_end"
        Column name for bin end in population_df
    total : str, default="total"
        Column name for population count/proportion

    Returns
    -------
    np.ndarray
        Array of weights (same length as data) with mean == 1 for non-zero weights

    Notes
    -----
    - Accepts integer or float ages
    - Handles boundary ages (include_lowest=True)
    - Leaves bins with zero sample untouched (weight=0 → later dropped)
    - Returns weights whose mean == 1 across *non-zero* weights
    """

    # 1. Make sure population bins are well-formed
    pop = (
        population_df[[range_start, range_end, total]]
        .sort_values(range_start)
        .reset_index(drop=True)
    )
    if (pop[range_end] <= pop[range_start]).any():
        raise ValueError("Each range_end must exceed range_start.")
    if (pop[range_start].iloc[1:].values < pop[range_end].iloc[:-1].values).any():
        raise ValueError("Age bins overlap or are not strictly increasing.")

    # 2. Bin edges for pd.cut  →  [..., last_end]  (right-inclusive)
    bin_edges = pop[range_start].tolist() + [pop[range_end].iloc[-1]]

    # Population proportions (works for % or counts)
    pop_prop = pop[total] / pop[total].sum()

    # 3. Assign each participant to a bin
    sample_bins = pd.cut(
        data[age_col],
        bins=bin_edges,
        right=True,  # include right edge
        include_lowest=True,  # include first left edge
        labels=pop.index,  # integer labels 0..n-1
    )

    # 4. Sample distribution
    sample_counts = sample_bins.value_counts(sort=False)
    sample_prop = sample_counts / sample_counts.sum()

    # 5. Weight lookup  (population / sample)  — careful with zeros
    weight_lookup = pop_prop / sample_prop.replace(0, np.nan)

    # 6. Map to rows; bins with zero sample → NaN weight → drop later
    w = sample_bins.map(weight_lookup).to_numpy()

    # Assign *zero* weight (instead of 1) to rows that were NaN
    w = np.where(np.isnan(w), 0.0, w)

    # 7. Rescale so mean(weight > 0) == 1
    positive = w > 0
    w[positive] = w[positive] / w[positive].mean()

    return w


def compute_joint_poststrat_weights(
    sample_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    *,
    age_col: str = "age_at_scan",
    sex_col: str = "sex",
    male_label=0,
    female_label=1,
    start_col: str = "range_start",
    end_col: str = "range_end",
    cap: Optional[float] = None,
    return_bin_table: bool = False,
) -> np.ndarray | Tuple[np.ndarray, pd.DataFrame]:
    """
    Computes joint post-stratification weights for Age and Sex so the sample
    matches an external census distribution.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Sample data with age and sex columns
    pop_df : pd.DataFrame
        Population data with [range_start, range_end, male, female] columns
    age_col : str, default="age_at_scan"
        Column name for age
    sex_col : str, default="sex"
        Column name for sex
    male_label : default=0
        Value used in sample_df[sex_col] for males
    female_label : default=1
        Value used in sample_df[sex_col] for females
    start_col : str, default="range_start"
        Column name for age bin start in pop_df
    end_col : str, default="range_end"
        Column name for age bin end in pop_df
    cap : float or None, optional
        Optional multiplier to truncate extreme weights
    return_bin_table : bool, default=False
        If True, returns (weights, bin_table_df) tuple

    Returns
    -------
    np.ndarray or tuple
        Weights array, or (weights, bin_table) if return_bin_table=True
    """
    # 1. Validation: Ensure all columns exist
    required_sample = [age_col, sex_col]
    required_pop = [start_col, end_col, "male", "female"]

    for col in required_sample:
        if col not in sample_df.columns:
            raise KeyError(
                f"Column '{col}' not found in sample_df. Available: {list(sample_df.columns)}"
            )
    for col in required_pop:
        if col not in pop_df.columns:
            raise KeyError(
                f"Column '{col}' not found in pop_df. Available: {list(pop_df.columns)}"
            )

    # 2. Tidy population table and convert to Long Format
    pop = pop_df.sort_values(start_col).copy()
    pop_long = pop.melt(
        id_vars=[start_col, end_col],
        value_vars=["male", "female"],
        var_name="sex_str",
        value_name="pop_val",
    )

    # Map 'male'/'female' strings to the labels in your sample (0 and 1)
    pop_long[sex_col] = pop_long["sex_str"].map({"male": male_label, "female": female_label})
    pop_long["prop_pop"] = pop_long["pop_val"] / pop_long["pop_val"].sum()

    # 3. Create Age Bins
    edges = sorted(list(pop[start_col].unique()) + [int(pop[end_col].max()) + 1])
    bin_labels = sorted(list(pop[start_col].unique()))

    sample = sample_df.copy()
    sample["age_bin_start"] = pd.cut(
        sample[age_col], bins=edges, labels=bin_labels, right=False  # standard for age: [start, end)
    )

    if sample["age_bin_start"].isna().any():
        print("Warning: Some participants fall outside the census age ranges.")

    # 4. Calculate Sample Proportions per (Age, Sex) cell
    sample_counts = (
        sample.groupby(["age_bin_start", sex_col], observed=False).size().reset_index(name="n_sample")
    )
    sample_counts["prop_sample"] = sample_counts["n_sample"] / sample_counts["n_sample"].sum()

    # Ensure types match for merging
    sample_counts["age_bin_start"] = sample_counts["age_bin_start"].astype(pop_long[start_col].dtype)

    # 5. Calculate Weight Factors (Population Proportion / Sample Proportion)
    weights_lookup = pd.merge(
        pop_long[[start_col, sex_col, "prop_pop"]],
        sample_counts,
        left_on=[start_col, sex_col],
        right_on=["age_bin_start", sex_col],
        how="left",
    )
    weights_lookup["weight_factor"] = weights_lookup["prop_pop"] / weights_lookup[
        "prop_sample"
    ].replace(0, np.nan)

    # 6. Map Weights back to individual sample rows
    sample["age_bin_start"] = sample["age_bin_start"].astype(pop_long[start_col].dtype)
    sample_with_weights = pd.merge(
        sample,
        weights_lookup[[start_col, sex_col, "weight_factor"]],
        left_on=["age_bin_start", sex_col],
        right_on=[start_col, sex_col],
        how="left",
    )

    w = sample_with_weights["weight_factor"].fillna(0).to_numpy()

    # 7. Apply optional cap and Rescale so mean weight = 1
    if cap is not None and cap > 0:
        mean_pos = w[w > 0].mean() if any(w > 0) else 1.0
        w = np.clip(w, 0, cap * mean_pos)

    if any(w > 0):
        w = w / w.mean()
    else:
        print("Error: No matches found between sample and census. Weights are all zero.")

    if return_bin_table:
        return w, weights_lookup

    return w


def calculate_weighting_stats(
    df: pd.DataFrame,
    census_df: pd.DataFrame,
    weight_col: str = "weight",
    age_col: str = "age_at_scan",
    sex_col: str = "sex",
) -> None:
    """
    Calculate and print weighting efficiency and representativeness statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Sample data with weight, age, and sex columns
    census_df : pd.DataFrame
        Census/population data with 'male', 'female', 'range_start', 'range_end' columns
    weight_col : str, default="weight"
        Column name for weights
    age_col : str, default="age_at_scan"
        Column name for age
    sex_col : str, default="sex"
        Column name for sex (assumes Female=1, Male=0)

    Notes
    -----
    Prints weighting efficiency (Kish's ESS) and representativeness metrics
    comparing raw sample, weighted sample, and census distributions.
    """
    # 1. Effective Sample Size (Kish's ESS)
    w = df[weight_col].values
    ess = (np.sum(w) ** 2) / np.sum(w**2)
    n_actual = len(df)

    # 2. Weighted Mean Age
    weighted_age = np.average(df[age_col], weights=df[weight_col])
    raw_age = df[age_col].mean()

    # 3. Weighted % Female (Assuming Female=1, Male=0)
    weighted_female = np.average(df[sex_col] == 1, weights=df[weight_col]) * 100
    raw_female = (df[sex_col] == 1).mean() * 100

    # 4. Census Benchmarks
    census_total = census_df["male"].sum() + census_df["female"].sum()
    census_female_pct = (census_df["female"].sum() / census_total) * 100
    # Approximate census age
    census_mid = (census_df["range_start"] + census_df["range_end"].replace(1000000, 95)) / 2
    census_age = np.average(census_mid, weights=(census_df["male"] + census_df["female"]))

    print("--- Weighting Efficiency ---")
    print(f"Actual N: {n_actual}")
    print(f"Effective N: {ess:.2f} ({ess/n_actual:.1%} efficiency)")
    print(f"Max Weight: {w.max():.2f}")

    print("\n--- Representativeness Check ---")
    print("Metric      | Census | Raw    | Weighted")
    print(f"Mean Age    | {census_age:.1f}   | {raw_age:.1f}   | {weighted_age:.1f}")
    print(f"% Female    | {census_female_pct:.1f}%  | {raw_female:.1f}%  | {weighted_female:.1f}%")


__all__ = [
    "compute_poststrat_weights",
    "assign_weights_by_population",
    "compute_joint_poststrat_weights",
    "calculate_weighting_stats",
]
