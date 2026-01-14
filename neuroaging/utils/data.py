"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from functools import reduce
from pathlib import Path


def load_metric_data(
    data_dir: Path,
    metrics: list[str],
    *,
    bad_subjects: list[str] | None = None,
    distribution_metric: str = "qfmean",
) -> dict[str, pd.DataFrame]:
    """
    Load and preprocess multiple brain metric datasets.

    Parameters
    ----------
    data_dir : Path
        Directory containing processed/{metric}.csv files.
    metrics : list of str
        Metric names to load (e.g., ['gm_vol', 'wm_vol', 'adc']).
    bad_subjects : list of str, optional
        Subject codes to exclude.
    distribution_metric : str
        Column name to use for distribution-based metrics (non-volume).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping metric name to preprocessed DataFrame.
    """
    if bad_subjects is None:
        bad_subjects = []

    data = {}
    for metric in metrics:
        df = pd.read_csv(
            data_dir / "processed" / f"{metric}.csv", index_col=0
        ).reset_index(drop=True)

        # Drop problematic subjects
        df = df[~df["subject_code"].isin(bad_subjects)]

        # Encode sex as numeric
        df["sex"] = df["sex"].map({"M": 0, "F": 1})

        # Rename value column to standardized name
        value_col = "volume" if "vol" in metric else distribution_metric
        if value_col in df.columns:
            df = df.rename(columns={value_col: "value"})

        # Ensure subject_code is zero-padded string
        df["subject_code"] = df["subject_code"].astype(str).str.zfill(4)

        data[metric] = df

    return data


def long_to_wide(
    long_df: pd.DataFrame,
    *,
    index: str = "subject_code",
    columns: str = "index",
    values: str = "value",
    missing_thresh: float = 0.8,
) -> pd.DataFrame:
    """
    Pivot a long metric table to wide format (subjects × regions).

    Parameters
    ----------
    long_df : DataFrame
        Long-format data with subject, region, and value columns.
    index : str
        Column to use as row index (subjects).
    columns : str
        Column to use as column headers (regions).
    values : str
        Column containing the values.
    missing_thresh : float
        Drop columns/rows with more than (1-thresh) missing values.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame.
    """
    wide = long_df.pivot_table(
        index=index, columns=columns, values=values, aggfunc="first"
    )

    # Drop regions with too many missing values
    thresh = int(missing_thresh * len(wide))
    wide = wide.dropna(axis=1, thresh=thresh)

    # Drop subjects with too many missing values
    thresh = int(missing_thresh * wide.shape[1])
    wide = wide.dropna(axis=0, thresh=thresh)

    return wide


def prep_metric_matrices(
    metric_dict: dict[str, pd.DataFrame],
    *,
    index: str = "subject_code",
    region_col: str = "index",
) -> tuple[dict[str, np.ndarray], pd.Series, pd.Series, pd.DataFrame]:
    """
    Convert multiple long-format metric DataFrames to aligned wide matrices.

    Parameters
    ----------
    metric_dict : dict[str, DataFrame]
        Mapping of metric name to long-format DataFrame.
    index : str
        Column name for subject identifier.
    region_col : str
        Column name for region identifier.

    Returns
    -------
    X_dict : dict[str, ndarray]
        Wide feature matrices aligned on the intersection of subjects.
    y : Series
        Age values for common subjects.
    w : Series
        Weights for common subjects (1.0 if not present).
    cov : DataFrame
        Covariates for common subjects.
    """
    # Convert each long DataFrame to wide
    wide_dict = {
        m: long_to_wide(df, index=index, columns=region_col)
        for m, df in metric_dict.items()
    }

    # Find subjects common to all metrics
    common_subs = reduce(
        pd.Index.intersection, [w.index for w in wide_dict.values()]
    )

    # Align all matrices to common subjects
    for m in wide_dict:
        wide_dict[m] = wide_dict[m].loc[common_subs]

    # Extract y, w, covariates from any of the long DataFrames
    ref_long = next(iter(metric_dict.values()))
    ref_meta = (
        ref_long.drop_duplicates(subset=index)
        .set_index(index)
        .loc[common_subs]
    )

    y = ref_meta["age_at_scan"]
    w = ref_meta.get("poststrat_weight", pd.Series(1.0, index=common_subs))
    cov = ref_meta[
        ref_meta.columns.difference(["age_at_scan", "poststrat_weight", "value"])
    ]

    # Convert wide DataFrames to numpy arrays
    X_dict = {m: w_df.to_numpy() for m, w_df in wide_dict.items()}

    return X_dict, y, w, cov
