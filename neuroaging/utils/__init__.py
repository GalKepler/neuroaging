"""Utility functions for neuroaging project."""

from neuroaging.utils.brain_age import (
    corrected_cross_val_predict,
    beheshti_correction as beheshti_bias_correction,
)
from neuroaging.utils.data import (
    load_metric_data,
    long_to_wide,
    prep_metric_matrices,
)
from neuroaging.utils.visualization import (
    COL_RAW,
    COL_REF,
    COL_WEIGHTED,
    CMAP_RAW,
    CMAP_WEIGHTED,
    METRIC_LABELS,
    savefig_nice,
    setup_plotting,
)
from neuroaging.utils.weights import (
    compute_joint_poststrat_weights,
    compute_poststrat_weights,
)

__all__ = [
    # brain_age
    "beheshti_bias_correction",
    "corrected_cross_val_predict",
    # data
    "load_metric_data",
    "long_to_wide",
    "prep_metric_matrices",
    # visualization
    "COL_RAW",
    "COL_REF",
    "COL_WEIGHTED",
    "CMAP_RAW",
    "CMAP_WEIGHTED",
    "METRIC_LABELS",
    "savefig_nice",
    "setup_plotting",
    # weights
    "compute_joint_poststrat_weights",
    "compute_poststrat_weights",
]
