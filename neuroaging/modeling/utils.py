"""
Model utility functions for extracting and formatting model results.

This module provides utilities for extracting statistical information from
fitted models and formatting values for publication.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def format_value_for_paper(
    value: float,
    decimal_places: int = 3,
    scientific_threshold_low: float = 1e-4,
    scientific_threshold_high: float = 1e5,
    scientific_precision: int = 2,
    latex_format: bool = False,
) -> str:
    """
    Format a float value for scientific papers using scientific notation when appropriate.

    Parameters
    ----------
    value : float
        The numerical value to format
    decimal_places : int, default=3
        Number of decimal places for non-scientific notation
    scientific_threshold_low : float, default=1e-4
        Values below this threshold use scientific notation
    scientific_threshold_high : float, default=1e5
        Values above this threshold use scientific notation
    scientific_precision : int, default=2
        Decimal places for mantissa in scientific notation
    latex_format : bool, default=False
        If True, formats for LaTeX (\\times 10^{{n}})

    Returns
    -------
    str
        Formatted string representation of the value

    Examples
    --------
    >>> format_value_for_paper(0.123456)
    '0.123'
    >>> format_value_for_paper(0.00001)
    '1.00 x 10^-5'
    >>> format_value_for_paper(150000)
    '1.50 x 10^5'
    """
    if pd.isna(value):
        return ""
    if value == 0.0:
        return "0." + "0" * decimal_places

    abs_val = abs(value)

    if (abs_val < scientific_threshold_low and abs_val != 0) or (abs_val >= scientific_threshold_high):
        # Scientific notation
        s_val = f"{value:.{scientific_precision}e}"
        parts = s_val.split("e")
        mantissa = parts[0]
        exponent = int(parts[1])

        if latex_format:
            return f"{mantissa} \\times 10^{{{{{exponent}}}}}"
        else:
            return f"{mantissa} x 10^{exponent}"
    else:
        # Fixed decimal
        return f"{value:.{decimal_places}f}"


def collect_relevant(
    data: Dict,
    relevant: list = ["Intercept", "age_at_scan", "I(age_at_scan ** 2)"],
) -> Dict[str, str]:
    """
    Extract and format relevant parameters from a dictionary.

    Parameters
    ----------
    data : dict
        Dictionary of parameter names to values
    relevant : list, default=["Intercept", "age_at_scan", "I(age_at_scan ** 2)"]
        List of parameter names to extract

    Returns
    -------
    dict
        Dictionary of relevant parameters with formatted values
    """
    return {key: format_value_for_paper(val) for key, val in data.items() if key in relevant}


def extract_model_info(
    model_results,
    model_type: str,
    linear_results=None,
) -> Dict:
    """
    Extract key statistical information from a statsmodels fitted model.

    Parameters
    ----------
    model_results : statsmodels RegressionResultsWrapper
        Fitted statsmodels results object
    model_type : str
        Type of model: 'linear' or 'quadratic'
    linear_results : statsmodels RegressionResultsWrapper, optional
        Fitted linear model (required for quadratic F-test comparison)

    Returns
    -------
    dict
        Dictionary containing formatted parameters, p-values, standard errors,
        R², AIC. For quadratic models with linear_results, also includes
        F-test statistics and preferred model.

    Examples
    --------
    >>> import statsmodels.formula.api as smf
    >>> # Fit models
    >>> lin_model = smf.ols("y ~ x", data=df).fit()
    >>> quad_model = smf.ols("y ~ x + I(x**2)", data=df).fit()
    >>> # Extract info
    >>> info = extract_model_info(quad_model, "quadratic", lin_model)
    >>> print(info['preferred_model'])
    'Quadratic'
    """
    info = {
        "params": collect_relevant(model_results.params.to_dict()),
        "pvalues": collect_relevant(model_results.pvalues.to_dict()),
        "bse": collect_relevant(model_results.bse.to_dict()),
        "rsquared": format_value_for_paper(model_results.rsquared_adj),
        "aic": model_results.aic,
    }

    if model_type == "quadratic" and linear_results is not None:
        try:
            f_stat, p_f_stat, _ = model_results.compare_f_test(linear_results)
            info["f_stat_q_vs_l"] = format_value_for_paper(f_stat)
            info["p_f_stat_q_vs_l"] = format_value_for_paper(p_f_stat)
            info["preferred_model"] = "Quadratic" if p_f_stat < 0.05 else "Linear"
        except Exception as e:
            print(f"Warning: Could not perform F-test comparison: {e}")
            info["f_stat_q_vs_l"] = np.nan
            info["p_f_stat_q_vs_l"] = np.nan
            info["preferred_model"] = "N/A"

    return info


__all__ = ["format_value_for_paper", "collect_relevant", "extract_model_info"]
