# Neuroaging

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Quantifying healthy brain aging using multimodal MRI metrics and statistical modeling.**

This project analyzes structural and diffusion MRI data to identify robust biomarkers of brain aging across the adult lifespan. It implements post-stratification weighting for population-representative estimates, regional age-association modeling, and stacked brain-age prediction.

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Analysis Notebooks](#analysis-notebooks)
  - [Demographics & Whole-Brain](#1-demographics--whole-brain)
  - [Regional Analyses](#2-regional-analyses)
  - [Age of Structural Transition (AST)](#3-age-of-structural-transition-ast)
  - [Stacking & Brain Age Gap (BAG)](#4-stacking--brain-age-gap-bag)
- [Figure Reference](#figure-reference)
- [Utility Modules](#utility-modules)
- [Reproducibility](#reproducibility)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/neuroaging.git
cd neuroaging

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the neuroaging package in development mode
pip install -e .
```

---

## Project Structure

```
neuroaging/
├── data/
│   ├── external/          # Atlas files, population data
│   ├── interim/           # Intermediate processed data
│   ├── processed/         # Final datasets (gm_vol.csv, adc.csv, etc.)
│   └── raw/               # Original data
│
├── neuroaging/            # Python package with reusable utilities
│   ├── config.py          # Project paths and configuration
│   ├── data/              # Data loading utilities
│   ├── modeling/          # Regional modeling, AST computation
│   ├── stats/             # Statistical functions, weights
│   ├── utils/             # Brain age, visualization, data utilities
│   └── visualization/     # Plotting configuration and helpers
│
├── notebooks/             # Analysis notebooks organized by module
│   ├── demographics_and_wholebrain/
│   ├── regional/
│   ├── ast/
│   └── stacking_and_bag/
│
├── reports/
│   └── figures/           # Generated figures for publication
│
├── README.md              # This file
├── QUICK_REFERENCE.md     # Quick reference for notebook updates
└── requirements.txt       # Python dependencies
```

---

## Analysis Notebooks

The analysis is organized into four modules, each generating specific figures for the manuscript. Run the notebooks in order, as later modules depend on outputs from earlier ones.

### 1. Demographics & Whole-Brain

**Location:** `notebooks/demographics_and_wholebrain/`

| Notebook | Description |
|----------|-------------|
| `wholebrain_and_demo_viz.ipynb` | Sample demographics, whole-brain age profiles |
| `wholebrain_redundancy.ipynb` | Metric intercorrelations and redundancy analysis |

**Figures Generated:**

| Figure | Description |
|--------|-------------|
| **Figure 1** | Age and sex distributions of the study cohort relative to the Israeli census |
| **Figure 2** | Whole-brain age-association profiles in GM volume and MD |
| **Supp. Figure S1** | Scatter plots: chronological age vs. diffusion-derived metrics |
| **Supp. Figure S2** | Global metric intercorrelations and redundancy analysis |
| **Supp. Figure S3** | Comparison of weighted and unweighted global age-associated profiles |

---

### 2. Regional Analyses

**Location:** `notebooks/regional/`

| Notebook | Description |
|----------|-------------|
| `regional_analyses.ipynb` | Regional age associations, surface visualizations |
| `clustering.ipynb` | Multimodal co-aging network analysis |

**Figures Generated:**

| Figure | Description |
|--------|-------------|
| **Figure 3** | Regional age associations for GM volume and MD |
| **Figure 5** | Multimodal "co-aging" networks |
| **Supp. Figure S4** | Stability of co-aging networks |

---

### 3. Age of Structural Transition (AST)

**Location:** `notebooks/ast/`

| Notebook | Description |
|----------|-------------|
| `ast.ipynb` | Compute and visualize regional AST across modalities |
| `ast_examples.ipynb` | Example visualizations and validation |

**Figures Generated:**

| Figure | Description |
|--------|-------------|
| **Figure 4** | Regional Age of Structural Transition (AST) across modalities |

---

### 4. Stacking & Brain Age Gap (BAG)

**Location:** `notebooks/stacking_and_bag/`

| Notebook | Description |
|----------|-------------|
| `brain_age_prediction.ipynb` | Train stacked brain-age models, feature importance |
| `bag_lifestyle_analysis.ipynb` | BAG vs. lifestyle factors, age-varying associations |

**Figures Generated:**

| Figure | Description |
|--------|-------------|
| **Figure 6** | Stacked vs. unimodal brain-age prediction performance |
| **Figure 7** | Age-dependent associations between brain-age gap (BAG) and phenotypes |
| **Supp. Figure S5** | Stability of age-dependent phenotype associations |

---

## Figure Reference

Quick reference for all manuscript figures and their source notebooks:

| Figure | Title | Notebook |
|--------|-------|----------|
| Fig. 1 | Cohort demographics | `demographics_and_wholebrain/wholebrain_and_demo_viz.ipynb` |
| Fig. 2 | Whole-brain age profiles | `demographics_and_wholebrain/wholebrain_and_demo_viz.ipynb` |
| Fig. 3 | Regional age associations | `regional/regional_analyses.ipynb` |
| Fig. 4 | Age of Structural Transition | `ast/ast.ipynb` |
| Fig. 5 | Co-aging networks | `regional/clustering.ipynb` |
| Fig. 6 | Brain-age prediction | `stacking_and_bag/brain_age_prediction.ipynb` |
| Fig. 7 | BAG-phenotype associations | `stacking_and_bag/bag_lifestyle_analysis.ipynb` |
| S1 | Diffusion scatter plots | `demographics_and_wholebrain/wholebrain_and_demo_viz.ipynb` |
| S2 | Metric redundancy | `demographics_and_wholebrain/wholebrain_redundancy.ipynb` |
| S3 | Weighted vs. unweighted | `demographics_and_wholebrain/wholebrain_and_demo_viz.ipynb` |
| S4 | Co-aging stability | `regional/clustering.ipynb` |
| S5 | BAG sensitivity analysis | `stacking_and_bag/bag_lifestyle_analysis.ipynb` |

---

## Utility Modules

The `neuroaging` package provides reusable functions for all notebooks:

### Visualization (`neuroaging.visualization`, `neuroaging.utils`)

```python
from neuroaging.utils import setup_plotting, savefig_nice, COL_WEIGHTED, COL_RAW

setup_plotting()  # Configure matplotlib for publication-quality figures
savefig_nice(fig, "output.png", dpi=300)
```

### Post-stratification Weights (`neuroaging.utils`, `neuroaging.stats`)

```python
from neuroaging.utils import compute_joint_poststrat_weights

weights, bin_table = compute_joint_poststrat_weights(
    sample_df, population_df,
    age_col="age_at_scan",
    sex_col="sex",
    cap=10,
    return_bin_table=True
)
```

### Regional Modeling (`neuroaging.modeling`)

```python
from neuroaging.modeling import fit_regional_models, apply_fdr_correction

results = fit_regional_models(
    data, parcels,
    metric_col="value",
    covariates=["age_at_scan", "sex", "tiv"],
    weight_col="weight"
)
apply_fdr_correction(results)
```

### Brain Age Prediction (`neuroaging.utils.brain_age`)

```python
from neuroaging.utils import corrected_cross_val_predict, beheshti_bias_correction

y_pred = corrected_cross_val_predict(
    model, X, y,
    cv=outer_cv,
    correction_func=beheshti_bias_correction
)
```

### Data Loading (`neuroaging.utils.data`)

```python
from neuroaging.utils import load_metric_data, prep_metric_matrices

data = load_metric_data(DATA_DIR, ["gm_vol", "adc", "fa"])
X_dict, y, w, cov = prep_metric_matrices(data)
```

---

## Reproducibility

### Running the Full Analysis

```bash
# 1. Demographics and whole-brain
jupyter nbconvert --execute notebooks/demographics_and_wholebrain/wholebrain_and_demo_viz.ipynb
jupyter nbconvert --execute notebooks/demographics_and_wholebrain/wholebrain_redundancy.ipynb

# 2. Regional analyses
jupyter nbconvert --execute notebooks/regional/regional_analyses.ipynb
jupyter nbconvert --execute notebooks/regional/clustering.ipynb

# 3. AST analysis
jupyter nbconvert --execute notebooks/ast/ast.ipynb

# 4. Brain age prediction and BAG analysis
jupyter nbconvert --execute notebooks/stacking_and_bag/brain_age_prediction.ipynb
jupyter nbconvert --execute notebooks/stacking_and_bag/bag_lifestyle_analysis.ipynb
```

### Data Requirements

The analysis requires the following data files in `data/processed/`:
- `gm_vol.csv`, `wm_vol.csv`, `csf_vol.csv` - Volume metrics
- `adc.csv`, `fa.csv`, `ad.csv`, `rd.csv` - Diffusion metrics
- `israel_population.csv` - Census data for post-stratification

Atlas files in `data/external/atlases/{atlas_name}/`:
- `parcels.csv` - Parcel information (labels, hemisphere, network)
- `atlas.nii.gz` - Volumetric atlas

### Environment

- Python 3.10+
- Key dependencies: numpy, pandas, scikit-learn, statsmodels, matplotlib, seaborn, nibabel, nilearn, surfplot

---

## License

See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code, please cite:

```bibtex
@article{neuroaging2025,
  title={Multimodal MRI biomarkers of healthy brain aging},
  author={...},
  journal={...},
  year={2025}
}
```
