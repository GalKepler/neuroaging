"""Tests for neuroaging.visualization module."""

import matplotlib.pyplot as plt
import pytest

from neuroaging.visualization import (
    CMAP_WEIGHTED,
    COL_RAW,
    configure_plotting,
    savefig_nice,
)


def test_configure_plotting():
    """Test matplotlib configuration works without errors."""
    configure_plotting()
    assert plt.rcParams["figure.dpi"] == 200
    # Font family might fall back to sans-serif if Calibri is not available
    assert isinstance(plt.rcParams["font.family"], list)


def test_color_constants():
    """Test color constants are defined."""
    assert COL_RAW == "#ffb300"
    assert CMAP_WEIGHTED is not None


def test_savefig_nice(tmp_path):
    """Test savefig_nice creates output file."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    output = tmp_path / "test.png"
    savefig_nice(fig, output)
    assert output.exists()
    plt.close(fig)
