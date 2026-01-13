"""Tests for neuroaging.modeling module."""

import numpy as np
import pandas as pd
import pytest

from neuroaging.modeling import RegionalModelResults, fit_regional_models


def test_fit_regional_models_smoke():
    """Smoke test for regional modeling with synthetic data."""
    # Create synthetic data
    np.random.seed(42)
    n_subjects = 100
    n_regions = 5

    data = pd.DataFrame(
        {
            "index": np.repeat(range(n_regions), n_subjects),
            "age_at_scan": np.tile(np.random.uniform(20, 70, n_subjects), n_regions),
            "sex": np.tile(np.random.choice([0, 1], n_subjects), n_regions),
            "value": np.random.randn(n_subjects * n_regions),
            "weight": np.ones(n_subjects * n_regions),
        }
    )

    parcels = pd.DataFrame({"index": range(n_regions)})

    # Fit models
    results = fit_regional_models(
        data, parcels, metric_col="value", region_col="index", covariates=["age_at_scan", "sex"]
    )

    # Basic assertions
    assert isinstance(results, RegionalModelResults)
    assert len(results.linear) == n_regions
    assert len(results.quadratic) == n_regions
    assert "beta1" in results.linear.columns
    assert "beta2" in results.quadratic.columns


def test_regional_model_results_attributes():
    """Test RegionalModelResults dataclass has expected attributes."""
    results = RegionalModelResults(
        linear=pd.DataFrame(),
        quadratic=pd.DataFrame(),
        comparison=pd.DataFrame(),
        centered=True,
        covariates=["age", "sex"],
    )
    assert results.centered is True
    assert results.covariates == ["age", "sex"]
