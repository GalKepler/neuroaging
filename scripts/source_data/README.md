# Source data generation

Regenerates the Nature Communications Source Data workbook (`reports/Source_Data.xlsx`)
for the 7 main figures and the 5 Supplementary Figures (S1-S5), straight from
the processed metric files - no participant-level data is ever written out,
only aggregated/model-derived results (fit curves, binned summaries, regional
stats, cluster assignments).

## Requirements

- The `neuroaging` package installed (repo venv).
- Access to the processed data directory. Scripts default to
  `/media/groot/Minerva/phd/neuroaging/data` (the drive the analysis
  notebooks currently read from). Override with `NEUROAGING_DATA_DIR` if
  your data lives elsewhere - **note**: the repo-local `data/` directory has
  been observed to diverge from the Minerva copy (different file hashes for
  at least `gm_vol.csv` and `adc.csv`), so don't assume it's equivalent.
- For Figure 5 and Figure 7 (column II), pre-existing aggregate CSVs from
  `notebooks/regional/clustering.ipynb` and
  `notebooks/stacking_and_bag/bag_lifestyle_analysis.ipynb` are reused
  rather than recomputed. Override their locations with
  `NEUROAGING_CLUSTERING_CSV` / `NEUROAGING_BAG_WINDOWS_DIR` if needed.

## Usage

```bash
python scripts/source_data/run_all.py
```

or run scripts individually (`build_fig1.py` ... `build_fig7.py`,
`build_figS1.py` ... `build_figS5.py`, then `assemble_workbook.py`). Each
`build_fig*.py` writes its CSVs to `reports/source_data/`;
`assemble_workbook.py` combines them into `reports/Source_Data.xlsx` with one
sheet per panel plus a README cover sheet.

`build_fig6.py` is by far the slowest of the main-figure scripts (7
whole-brain Ridge grid searches + 454 parcel-wise grid searches +
permutation importance with `n_repeats=100`) - budget 20-40+ minutes
depending on the machine.

`build_figS4.py` (clustering sensitivity: weighted vs. unweighted GMM
model-selection sweep, K=2..20, `n_init=1000` per fit) is comparably slow -
budget 30-60+ minutes. It also expects the weighted/unweighted cluster
assignment CSVs used by `build_fig5.py` (`NEUROAGING_CLUSTERING_CSV` /
`NEUROAGING_UNWEIGHTED_CLUSTERING_CSV`) to already exist for its spatial
topography and Dice-stability panels - it reuses those cached assignments
rather than refitting the final model, to stay consistent with the
manuscript figure. `build_figS5.py` reuses the precomputed sliding-window
parameter-sweep CSVs (`NEUROAGING_SLIDING_WINDOW_GRID_DIR`, default
`.../fig_BAG/N{50,75,100,150,200}_S{5,10,15,20}/`) and is fast.

## Known gaps (see also the README sheet in the generated workbook)

- **Figure 3/4 MD quadratic-preferred count**: rerunning against the data
  mounted at generation time gave 377/454 MD regions passing the dual
  threshold (FDR q<0.05 AND ΔAIC<-15), vs the manuscript's 414/454. GM
  volume matched exactly (139/454), and the FDR-only count for MD (414)
  also matched exactly - the gap is ~37 regions with ΔAIC between about
  -14 and -2, consistent with minor drift in the processed files since the
  run that produced the submitted numbers. Re-check against the exact data
  snapshot used for submission before treating this as final.
- **Figure 7 column I** (global weighted BAG~phenotype models, with/without
  age interaction) needs `~/Projects/PlasticityHub/sessions.csv` and
  per-subject FreeSurfer stats under
  `/media/storage/yalab-dev/derivatives/freesurfer/` - override with
  `NEUROAGING_PLASTICITYHUB_SESSIONS_CSV` / `NEUROAGING_FREESURFER_DIR` if
  yours live elsewhere. Now reproduced in full alongside column II.
