"""
Statistical utilities for neuroaging project.

This package provides statistical functions used across analyses, including
post-stratification weighting and hypothesis testing utilities.
"""

from neuroaging.utils.utils import compute_poststrat_weights
from .weights import (
    assign_weights_by_population,
    compute_joint_poststrat_weights,
    calculate_weighting_stats,
)

__all__ = [
    "compute_poststrat_weights",
    "assign_weights_by_population",
    "compute_joint_poststrat_weights",
    "calculate_weighting_stats",
]
