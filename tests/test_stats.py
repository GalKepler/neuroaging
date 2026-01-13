"""Tests for neuroaging.stats module."""

import pytest

from neuroaging.stats import compute_poststrat_weights


def test_compute_poststrat_weights_import():
    """Test that compute_poststrat_weights can be imported."""
    assert callable(compute_poststrat_weights)
