"""Post-stratification weight computation utilities."""

import numpy as np
import pandas as pd


def compute_poststrat_weights(
    sample_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    *,
    age_col: str = "age_at_scan",
    start_col: str = "range_start",
    end_col: str = "range_end",
    pop_total_col: str = "total",
    cap: float | None = None,
    return_bin_table: bool = False,
):
    """
    Post-stratification weights so that the age distribution of *sample_df*
    matches an external population distribution supplied in *pop_df*.

    Parameters
    ----------
    sample_df : DataFrame with an ``age_col`` column (years; int or float).
    pop_df    : DataFrame with columns [start_col, end_col, pop_total_col].
    cap       : Optional float. Weights truncated at ``cap × mean(weight)``.
    return_bin_table : If True, also return a per-bin summary DataFrame.

    Returns
    -------
    weights   : 1-D numpy array aligned with ``sample_df.index``.
    bin_table : (optional) tidy per-bin summary.
    """
    pop = (
        pop_df[[start_col, end_col, pop_total_col]]
        .dropna()
        .astype({start_col: int, end_col: int, pop_total_col: float})
        .sort_values(start_col)
        .reset_index(drop=True)
    )

    if (pop[end_col] <= pop[start_col]).any():
        raise ValueError("Each range_end must exceed range_start.")

    edges = pop[start_col].tolist() + [pop[end_col].iloc[-1]]

    s_bins = pd.cut(
        sample_df[age_col],
        bins=edges,
        right=True,
        include_lowest=True,
        labels=pop.index,
    )

    n_sample = s_bins.value_counts(sort=False).sort_index()
    prop_sample = n_sample / n_sample.sum()
    prop_pop = pop[pop_total_col] / pop[pop_total_col].sum()

    weight_factor = prop_pop / prop_sample.replace(0, np.nan)

    w = s_bins.map(weight_factor).astype(float).to_numpy()
    w = np.where(np.isnan(w), 0.0, w)

    if cap is not None and cap > 0:
        mean_pos = w[w > 0].mean()
        w = np.clip(w, 0, cap * mean_pos)

    positive = w > 0
    w[positive] = w[positive] / w[positive].mean()

    if return_bin_table:
        bin_tbl = pd.DataFrame(
            {
                "n_sample": n_sample,
                "n_pop": prop_pop * n_sample.sum(),
                "prop_sample": prop_sample,
                "prop_pop": prop_pop,
                "weight_factor": weight_factor,
            }
        )
        return w, bin_tbl

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
    cap: float | None = None,
    return_bin_table: bool = False,
):
    """
    Compute joint post-stratification weights for Age and Sex so the sample
    matches an external census distribution.

    Parameters
    ----------
    sample_df : DataFrame with age and sex columns.
    pop_df    : DataFrame with [range_start, range_end, male, female].
    male_label/female_label : Values used in sample_df['sex'] (e.g., 0 and 1).
    cap       : Optional multiplier to truncate extreme weights.
    return_bin_table : If True, return (weights, bin_table).

    Returns
    -------
    weights   : 1-D numpy array aligned with sample_df.index.
    bin_table : (optional) tidy per-cell summary.
    """
    required_sample = [age_col, sex_col]
    required_pop = [start_col, end_col, "male", "female"]

    for col in required_sample:
        if col not in sample_df.columns:
            raise KeyError(f"Column '{col}' not found in sample_df.")
    for col in required_pop:
        if col not in pop_df.columns:
            raise KeyError(f"Column '{col}' not found in pop_df.")

    pop = pop_df.sort_values(start_col).copy()
    pop_long = pop.melt(
        id_vars=[start_col, end_col],
        value_vars=["male", "female"],
        var_name="sex_str",
        value_name="pop_val",
    )

    pop_long[sex_col] = pop_long["sex_str"].map(
        {"male": male_label, "female": female_label}
    )
    pop_long["prop_pop"] = pop_long["pop_val"] / pop_long["pop_val"].sum()

    edges = sorted(list(pop[start_col].unique()) + [int(pop[end_col].max()) + 1])
    bin_labels = sorted(list(pop[start_col].unique()))

    sample = sample_df.copy()
    sample["age_bin_start"] = pd.cut(
        sample[age_col], bins=edges, labels=bin_labels, right=False
    )

    sample_counts = (
        sample.groupby(["age_bin_start", sex_col], observed=True).size().reset_index(name="n_sample")
    )
    sample_counts["prop_sample"] = sample_counts["n_sample"] / sample_counts["n_sample"].sum()

    pop_long_renamed = pop_long.rename(columns={start_col: "age_bin_start"})
    merged = sample_counts.merge(
        pop_long_renamed[["age_bin_start", sex_col, "prop_pop"]],
        on=["age_bin_start", sex_col],
        how="left",
    )

    merged["weight_factor"] = merged["prop_pop"] / merged["prop_sample"].replace(0, np.nan)

    sample_indexed = sample.set_index(["age_bin_start", sex_col])
    merged_indexed = merged.set_index(["age_bin_start", sex_col])["weight_factor"]
    sample["weight_raw"] = sample_indexed.index.map(merged_indexed.to_dict())
    w = sample["weight_raw"].to_numpy()
    w = np.where(np.isnan(w), 0.0, w)

    if cap is not None and cap > 0:
        mean_pos = w[w > 0].mean()
        w = np.clip(w, 0, cap * mean_pos)

    positive = w > 0
    if positive.any():
        w[positive] = w[positive] / w[positive].mean()

    if return_bin_table:
        return w, merged
    return w
