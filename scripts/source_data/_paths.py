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

DATA_DIR = Path(os.environ.get("NEUROAGING_DATA_DIR", "/media/storage/phd/neuroaging/data"))

OUT_DIR = REPO_ROOT / "reports" / "source_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Precomputed aggregate files this pipeline reuses instead of recomputing
CLUSTERING_CSV = Path(
    os.environ.get(
        "NEUROAGING_CLUSTERING_CSV",
        "/media/storage/phd/neuroaging/figures/revision/fig2_clustering/weighted/region_clusters.csv",
    )
)
UNWEIGHTED_CLUSTERING_CSV = Path(
    os.environ.get(
        "NEUROAGING_UNWEIGHTED_CLUSTERING_CSV",
        "/media/storage/phd/neuroaging/figures/revision/fig2_clustering/unweighted/region_clusters.csv",
    )
)
SLIDING_WINDOW_GRID_DIR = Path(
    os.environ.get(
        "NEUROAGING_SLIDING_WINDOW_GRID_DIR",
        "/media/storage/phd/neuroaging/figures/revision/fig_BAG",
    )
)
BAG_WINDOWS_DIR = Path(
    os.environ.get(
        "NEUROAGING_BAG_WINDOWS_DIR",
        "/media/storage/phd/neuroaging/figures/revision/fig_BAG/N100_S10",
    )
)

# Inputs for Figure 7 column I (global weighted BAG~phenotype models)
BAG_PREDICTIONS_CSV = Path(
    os.environ.get(
        "NEUROAGING_BAG_PREDICTIONS_CSV",
        str(REPO_ROOT / "notebooks" / "stacking_and_bag" / "BAG_data.csv"),
    )
)
PLASTICITYHUB_SESSIONS_CSV = Path(
    os.environ.get("NEUROAGING_PLASTICITYHUB_SESSIONS_CSV", "~/Projects/PlasticityHub/sessions.csv")
).expanduser()
FREESURFER_DIR = Path(
    os.environ.get("NEUROAGING_FREESURFER_DIR", "/media/storage/yalab-dev/derivatives/freesurfer")
)
