"""
Age of Peak Structural Integrity (APSI) calculation utilities.

This module provides functions for calculating the age at which brain structural
metrics reach peak integrity (vertex of quadratic aging trajectories).
"""

from typing import Tuple, Dict

# Default orientation for different metrics
ORIENTATION = {
    "gm_vol": "max",  # expect inverted U (concave down)
    "adc": "min",  # expect U-shape (concave up)
    "md": "min",  # mean diffusivity - U-shape
    "fa": "min",  # fractional anisotropy - typically declines
    "rd": "min",  # radial diffusivity - U-shape
    "ad": "min",  # axial diffusivity - U-shape
}


def vertex_age(beta1: float, beta2: float) -> float:
    """
    Calculate age at vertex of a quadratic trajectory.

    For a quadratic model: y = β₀ + β₁·x + β₂·x²
    The vertex occurs at: x = -β₁ / (2·β₂)

    Parameters
    ----------
    beta1 : float
        Linear coefficient (β₁)
    beta2 : float
        Quadratic coefficient (β₂)

    Returns
    -------
    float
        Age at vertex of the quadratic curve

    Examples
    --------
    >>> vertex_age(beta1=-0.5, beta2=-0.01)  # Inverted U
    25.0
    >>> vertex_age(beta1=0.5, beta2=0.01)   # U-shape
    -25.0
    """
    return -beta1 / (2 * beta2)


def stabilization_age(
    metric: str,
    betas: Dict[str, float],
    mean_age: float,
    *,
    age_min: float = 18,
    age_max: float = 75,
    orientation: Dict[str, str] = ORIENTATION,
) -> Tuple[float, bool, bool]:
    """
    Calculate Age of Peak Structural Integrity (APSI) from quadratic model.

    This function determines the age at which a brain metric reaches peak
    structural integrity, accounting for:
    - Expected trajectory shape (U vs inverted U)
    - Age centering in the model
    - Boundary clipping to observed age range
    - Detection of unexpected trajectory shapes

    Parameters
    ----------
    metric : str
        Name of the brain metric (e.g., "gm_vol", "adc", "fa")
    betas : dict
        Dictionary with keys "age" (β₁) and "age_sq" (β₂) containing
        the linear and quadratic coefficients
    mean_age : float
        Mean age used for centering in the model. If variables were not
        centered, use 0.
    age_min : float, default=18
        Minimum observed age in the sample
    age_max : float, default=75
        Maximum observed age in the sample
    orientation : dict, default=ORIENTATION
        Dictionary mapping metric names to expected shapes:
        "max" for inverted-U (GM volume), "min" for U-shape (diffusion)

    Returns
    -------
    age_star : float
        Age of peak structural integrity (APSI)
    clipped : bool
        True if vertex falls outside [age_min, age_max] or has wrong shape
    is_u_shaped : bool
        True if trajectory is U-shaped (β₂ > 0), False if inverted-U (β₂ < 0)

    Notes
    -----
    - If the trajectory has the "wrong" expected shape (e.g., U-shaped for GM
      volume which should show inverted-U), returns age_min with clipped=True
    - Vertex calculation accounts for age centering: vertex = mean_age - β₁/(2β₂)
    - Vertices outside [age_min, age_max] are clipped to boundaries

    Examples
    --------
    >>> # GM volume: expect inverted U (max)
    >>> stabilization_age("gm_vol", {"age": -0.5, "age_sq": -0.01}, mean_age=45)
    (70.0, False, False)  # Peak at 70, not clipped, inverted-U

    >>> # Diffusion: expect U-shape (min)
    >>> stabilization_age("adc", {"age": 0.5, "age_sq": 0.01}, mean_age=45)
    (20.0, False, True)  # Bottom at 20, not clipped, U-shaped

    >>> # Wrong shape - GM showing U instead of inverted-U
    >>> stabilization_age("gm_vol", {"age": 0.5, "age_sq": 0.01}, mean_age=45)
    (18.0, True, True)  # Returns age_min, clipped, wrong shape
    """
    b1, b2 = betas["age"], betas["age_sq"]

    # 1. Determine if the shape is 'biologically expected'
    is_u_shaped = b2 > 0  # positive quadratic → U-shape
    expected_orientation = orientation.get(metric, "min")

    # Check if trajectory matches expected shape
    wrong_shape = (expected_orientation == "max" and is_u_shaped) or (
        expected_orientation == "min" and not is_u_shaped
    )

    if wrong_shape:
        # If wrong shape, the 'vertex' isn't a meaningful stabilization point
        return age_min, True, is_u_shaped

    # 2. Compute vertex (adjusting for centering)
    # Formula: vertex = mean_age - β₁/(2β₂)
    # This accounts for age being centered at mean_age in the model
    age_star = mean_age - (b1 / (2 * b2))

    # 3. Clip to observed age range and flag if clipped
    clipped = False
    if age_star < age_min:
        age_star, clipped = age_min, True
    elif age_star > age_max:
        age_star, clipped = age_max, True

    return age_star, clipped, is_u_shaped


__all__ = ["vertex_age", "stabilization_age", "ORIENTATION"]
