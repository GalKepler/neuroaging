"""
Regional brain aging modeling utilities.

This module provides functions for fitting linear and quadratic weighted least squares
(WLS) models to regional brain metrics across the lifespan. It eliminates ~100 lines
of duplicated modeling code in 15+ notebooks.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RegionalModelResults:
    """
    Container for regional modeling results.

    Attributes
    ----------
    linear : pd.DataFrame
        Results from linear models (one row per region) with columns:
        model, pvalue, r2, r2_adj, beta1, pvalue_beta1, n
    quadratic : pd.DataFrame
        Results from quadratic models with additional columns:
        beta2, pvalue_beta2
    comparison : pd.DataFrame
        Model comparison statistics with columns:
        f, pvalue, log(f), AIC(lin), AIC(quad), delta_AIC, BIC(lin), BIC(quad), delta_BIC
    centered : bool
        Whether age was centered before fitting
    covariates : List[str]
        List of covariate names used in models
    """

    linear: pd.DataFrame
    quadratic: pd.DataFrame
    comparison: pd.DataFrame
    centered: bool
    covariates: List[str]


def fit_regional_models(
    data: pd.DataFrame,
    parcels: pd.DataFrame,
    *,
    metric_col: str,
    region_col: str = "index",
    age_col: str = "age_at_scan",
    covariates: Optional[List[str]] = None,
    center_variables: bool = True,
    weight_col: Optional[str] = "weight",
) -> RegionalModelResults:
    """
    Fit linear and quadratic WLS models to each brain region.

    This function fits both linear and quadratic models to regional data,
    performs F-tests to compare models, and calculates AIC/BIC for model selection.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data with columns [region_col, age_col, metric_col, weight_col].
        Each row represents one subject's measurement for one region.
    parcels : pd.DataFrame
        Region metadata (one row per region). Used as template for results.
        Must contain a column matching region_col.
    metric_col : str
        Column name for the brain metric to model (e.g., "value", "gm_vol")
    region_col : str, default="index"
        Column name identifying regions
    age_col : str, default="age_at_scan"
        Column name for age variable
    covariates : list of str, optional
        Additional covariates to include (default: [age_col, "sex"]).
        Age will be included automatically.
    center_variables : bool, default=True
        Whether to center age before fitting. If True, creates "age_c" column
        and uses it in formulas. Centering improves numerical stability and
        makes the vertex calculation simpler.
    weight_col : str or None, default="weight"
        Column for post-stratification weights. If None or not present in data,
        uses OLS instead of WLS.

    Returns
    -------
    RegionalModelResults
        Dataclass containing three DataFrames (linear, quadratic, comparison)
        with model fits, p-values, R², betas, and comparison statistics for
        each region.

    Examples
    --------
    >>> from neuroaging.modeling import fit_regional_models, apply_fdr_correction
    >>> # Assuming data and parcels are loaded
    >>> results = fit_regional_models(
    ...     data,
    ...     parcels,
    ...     metric_col="value",
    ...     covariates=["age_at_scan", "sex", "tiv"],
    ...     center_variables=True,
    ...     weight_col="weight"
    ... )
    >>> apply_fdr_correction(results)
    >>> # Access results
    >>> lin_w = results.linear
    >>> quad_w = results.quadratic
    >>> comparison = results.comparison

    Notes
    -----
    - Linear model: value ~ age + sex + ...
    - Quadratic model: value ~ age + age² + sex + ...
    - F-test compares nested models (quadratic vs linear)
    - AIC/BIC used for model selection (lower is better)
    """
    if covariates is None:
        covariates = [age_col, "sex"]

    # Prepare age column
    if center_variables:
        age_mean = data[age_col].mean()
        data = data.copy()
        data["age_c"] = data[age_col] - age_mean
        age_var = "age_c"
    else:
        age_var = age_col
        age_mean = 0

    # Initialize result dataframes
    lin_results = parcels.copy()
    quad_results = parcels.copy()
    comparison = parcels.copy()

    # Add columns for storing results
    for df in [lin_results, quad_results]:
        df["model"] = None
        df["pvalue"] = np.nan
        df["r2"] = np.nan
        df["r2_adj"] = np.nan
        df["beta1"] = np.nan
        df["pvalue_beta1"] = np.nan
        df["n"] = np.nan

    quad_results["beta2"] = np.nan
    quad_results["pvalue_beta2"] = np.nan

    comparison["f"] = np.nan
    comparison["pvalue"] = np.nan
    comparison["log(f)"] = np.nan
    comparison["n"] = np.nan
    comparison["AIC(lin)"] = np.nan
    comparison["AIC(quad)"] = np.nan
    comparison["delta_AIC"] = np.nan
    comparison["BIC(lin)"] = np.nan
    comparison["BIC(quad)"] = np.nan
    comparison["delta_BIC"] = np.nan

    # Fit models for each region
    for i, row in parcels.iterrows():
        region_data = data[data[region_col] == row[region_col]].copy()

        if len(region_data) == 0:
            continue

        # Rename metric column to "value" for formula
        region_data = region_data.rename(columns={metric_col: "value"})

        # Build formulas
        cov_list = [c if c != age_col else age_var for c in covariates]
        formula_lin = f"value ~ {' + '.join(cov_list)}"
        formula_quad = formula_lin + f" + I({age_var} ** 2)"

        try:
            # Fit models
            if weight_col and weight_col in region_data.columns:
                lin_model = smf.wls(
                    formula_lin, data=region_data, weights=region_data[weight_col]
                ).fit()
                quad_model = smf.wls(
                    formula_quad, data=region_data, weights=region_data[weight_col]
                ).fit()
            else:
                lin_model = smf.ols(formula_lin, data=region_data).fit()
                quad_model = smf.ols(formula_quad, data=region_data).fit()

            # Store linear results
            lin_results.loc[i, "model"] = lin_model
            lin_results.loc[i, "pvalue"] = lin_model.f_pvalue
            lin_results.loc[i, "r2"] = lin_model.rsquared
            lin_results.loc[i, "r2_adj"] = lin_model.rsquared_adj
            lin_results.loc[i, "beta1"] = lin_model.params.get(age_var, np.nan)
            lin_results.loc[i, "pvalue_beta1"] = lin_model.pvalues.get(age_var, np.nan)
            lin_results.loc[i, "n"] = lin_model.nobs

            # Store quadratic results
            quad_results.loc[i, "model"] = quad_model
            quad_results.loc[i, "pvalue"] = quad_model.f_pvalue
            quad_results.loc[i, "r2"] = quad_model.rsquared
            quad_results.loc[i, "r2_adj"] = quad_model.rsquared_adj
            quad_results.loc[i, "beta1"] = quad_model.params.get(age_var, np.nan)
            quad_results.loc[i, "pvalue_beta1"] = quad_model.pvalues.get(age_var, np.nan)
            quad_results.loc[i, "beta2"] = quad_model.params.get(f"I({age_var} ** 2)", np.nan)
            quad_results.loc[i, "pvalue_beta2"] = quad_model.pvalues.get(
                f"I({age_var} ** 2)", np.nan
            )
            quad_results.loc[i, "n"] = quad_model.nobs

            # F-test comparison
            f, p, _ = quad_model.compare_f_test(lin_model)
            comparison.loc[i, "f"] = f
            comparison.loc[i, "pvalue"] = p
            comparison.loc[i, "log(f)"] = np.log(f) if f > 0 else np.nan
            comparison.loc[i, "n"] = lin_model.nobs
            comparison.loc[i, "AIC(lin)"] = lin_model.aic
            comparison.loc[i, "AIC(quad)"] = quad_model.aic
            comparison.loc[i, "delta_AIC"] = quad_model.aic - lin_model.aic
            comparison.loc[i, "BIC(lin)"] = lin_model.bic
            comparison.loc[i, "BIC(quad)"] = quad_model.bic
            comparison.loc[i, "delta_BIC"] = quad_model.bic - lin_model.bic

        except Exception as e:
            # Log error but continue with other regions
            print(f"Warning: Could not fit models for region {row[region_col]}: {e}")
            continue

    return RegionalModelResults(
        linear=lin_results,
        quadratic=quad_results,
        comparison=comparison,
        centered=center_variables,
        covariates=covariates,
    )


def apply_fdr_correction(
    results: RegionalModelResults, method: str = "fdr_bh", alpha: float = 0.05
):
    """
    Apply FDR correction to p-values in model results (in-place).

    This function adds new columns with "_corrected" suffix containing
    FDR-adjusted p-values for all p-value columns in the results.

    Parameters
    ----------
    results : RegionalModelResults
        Results object from fit_regional_models()
    method : str, default="fdr_bh"
        Multiple testing correction method. Options:
        - "fdr_bh": Benjamini-Hochberg FDR
        - "bonferroni": Bonferroni correction
        - "fdr_by": Benjamini-Yekutieli FDR
    alpha : float, default=0.05
        Family-wise error rate

    Returns
    -------
    None
        Modifies results DataFrames in-place by adding "_corrected" columns

    Examples
    --------
    >>> results = fit_regional_models(data, parcels, metric_col="value")
    >>> apply_fdr_correction(results)
    >>> # Now results.linear has "pvalue_corrected", "pvalue_beta1_corrected" etc.
    >>> sig_regions = results.linear[results.linear["pvalue_corrected"] < 0.05]

    Notes
    -----
    FDR correction is applied separately to each DataFrame (linear, quadratic,
    comparison) and separately to each p-value column within them.
    """
    from statsmodels.stats.multitest import multipletests

    for df_name in ["linear", "quadratic", "comparison"]:
        df = getattr(results, df_name)
        for pcol in df.columns:
            if "pvalue" in pcol and "corrected" not in pcol:
                # Skip if all NaN
                if not df[pcol].notna().any():
                    continue

                # Apply FDR correction
                reject, pvals_corrected, _, _ = multipletests(
                    df[pcol].fillna(1.0), alpha=alpha, method=method
                )
                df[f"{pcol}_corrected"] = pvals_corrected


__all__ = ["fit_regional_models", "apply_fdr_correction", "RegionalModelResults"]
