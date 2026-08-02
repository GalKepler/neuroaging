# Source data generation

Regenerates the Nature Communications Source Data workbook (`reports/Source_Data.xlsx`)
for the 7 main figures, straight from the processed metric files - no
participant-level data is ever written out, only aggregated/model-derived
results (fit curves, binned summaries, regional stats, cluster assignments).

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

or run scripts individually (`build_fig1.py` ... `build_fig7.py`, then
`assemble_workbook.py`). Each `build_fig*.py` writes its CSVs to
`reports/source_data/`; `assemble_workbook.py` combines them into
`reports/Source_Data.xlsx` with one sheet per panel plus a README cover
sheet.

`build_fig6.py` is by far the slowest (7 whole-brain Ridge grid searches +
454 parcel-wise grid searches + permutation importance with
`n_repeats=100`) - budget 20-40+ minutes depending on the machine.

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
  age interaction) is not reproduced by `build_fig7.py` - it needs
  `~/Projects/plasticityhub/sessions.csv` and per-subject FreeSurfer stats
  under `/media/storage/yalab-dev/derivatives/freesurfer/`, which weren't
  available on the machine this pipeline was built on. Column II (the
  sliding-window analysis) is included in full.
