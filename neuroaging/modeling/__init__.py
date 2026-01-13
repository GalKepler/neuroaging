"""
Modeling utilities for neuroaging project.

This package provides functions for fitting statistical models to brain aging data,
including regional aging models, brain age prediction, and APSI calculations.
"""

from .regional import RegionalModelResults, apply_fdr_correction, fit_regional_models
from .apsi import vertex_age, stabilization_age, ORIENTATION
from .utils import format_value_for_paper, collect_relevant, extract_model_info

__all__ = [
    # Regional modeling
    "fit_regional_models",
    "apply_fdr_correction",
    "RegionalModelResults",
    # APSI
    "vertex_age",
    "stabilization_age",
    "ORIENTATION",
    # Utils
    "format_value_for_paper",
    "collect_relevant",
    "extract_model_info",
]
