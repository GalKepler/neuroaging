# Stacking & Brain Age Gap Analysis

This module implements stacked brain-age prediction and analyzes relationships between Brain Age Gap (BAG) and lifestyle/health factors.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `brain_age_prediction.ipynb` | Train stacked brain-age models using multiple MRI metrics, evaluate performance, compute feature importance |
| `bag_lifestyle_analysis.ipynb` | Analyze BAG associations with lifestyle factors, rolling window analysis for age-varying effects |

## Figures Generated

| Figure | Description |
|--------|-------------|
| **Figure 6** | Stacked vs. unimodal brain-age prediction performance |
| **Figure 7** | Age-dependent associations between brain-age gap (BAG) and phenotypes |
| **Supp. Figure S5** | Stability of age-dependent phenotype associations |

## Dependencies

Run `brain_age_prediction.ipynb` first to generate `BAG_data.csv`, which is required by `bag_lifestyle_analysis.ipynb`.

## Key Outputs

- `BAG_data.csv`: Per-subject BAG predictions for all metrics and stacked model
- Performance comparison plots
- Feature importance surface maps (SHAP + permutation)
- Age-varying association plots with sensitivity analysis
