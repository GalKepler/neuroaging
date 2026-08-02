"""Shared paths for the source-data generation scripts in this directory.

DATA_DIR defaults to the external drive the analysis notebooks read from
(the up-to-date processed metrics live there, not in the gitignored
repo-local data/ directory - see README.md in this folder for why that
matters). Override with the NEUROAGING_DATA_DIR env var if your data lives
elsewhere.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("NEUROAGING_DATA_DIR", "/media/groot/Minerva/phd/neuroaging/data"))

OUT_DIR = REPO_ROOT / "reports" / "source_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Precomputed aggregate files this pipeline reuses instead of recomputing
CLUSTERING_CSV = Path(
    os.environ.get(
        "NEUROAGING_CLUSTERING_CSV",
        "/media/groot/Minerva/phd/neuroaging/figures/revision/fig2_clustering/weighted/region_clusters.csv",
    )
)
BAG_WINDOWS_DIR = Path(
    os.environ.get(
        "NEUROAGING_BAG_WINDOWS_DIR",
        "/media/groot/Minerva/phd/neuroaging/figures/revision/fig_BAG/N100_S10",
    )
)
