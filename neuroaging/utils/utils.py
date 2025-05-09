import mpmath
import numpy as np


def high_precision_pvalue(r2: float, n: int, dps=3):
    """
    Calculate high-precision p-value for an F-statistic using mpmath.

    Parameters:
    - r2: Adjusted R-squared value from the regression model.
    - n: Number of observations.
    - dps: Decimal places of precision for mpmath calculations.

    Returns:
    - High-precision p-value (mpmath.mpf type)
    """
    mpmath.mp.dps = dps
    x = (-np.sqrt(r2) + 1) / 2  # shift per `loc=-1`, scale per `scale=2`
    pval_mpmath = 2 * mpmath.mp.betainc(n / 2 - 1, n / 2 - 1, 0, x, regularized=True)

    return pval_mpmath
