# Quick Reference: Updating Notebooks

## 🔄 Copy-Paste These Imports

### For ALL Notebooks (replaces ~100 lines):
```python
from neuroaging.visualization import (
    configure_plotting, savefig_nice,
    COL_RAW, COL_WEIGHTED, COL_REF, COL_CENSUS,
    CMAP_RAW, CMAP_WEIGHTED
)
configure_plotting()
```

### For Regional Modeling (fig2, fig3 notebooks):
```python
from neuroaging.modeling import fit_regional_models, apply_fdr_correction

results = fit_regional_models(
    data,
    parcels,
    metric_col="value",
    region_col="index",
    age_col="age_at_scan",
    covariates=["age_at_scan", "sex", "tiv"],
    center_variables=True,
    weight_col="weight"
)

apply_fdr_correction(results)

# Access results
lin_w = results.linear
quad_w = results.quadratic
w_compare = results.comparison
```

### For Weight Calculations (19+ notebooks):
```python
from neuroaging.stats import compute_poststrat_weights
```

## ✅ Checklist

- [ ] Backup notebook before editing
- [ ] Replace visualization config with import
- [ ] Replace regional modeling loop (if applicable)
- [ ] Replace weight function definition with import
- [ ] Run entire notebook
- [ ] Compare key outputs (betas, p-values, figures)
- [ ] Commit changes

## 🧪 Quick Validation

```bash
# Test the refactored modules
python validate_refactoring.py

# Run test suite
pytest tests/ -v

# Check imports work
python -c "from neuroaging.visualization import configure_plotting; configure_plotting()"
python -c "from neuroaging.modeling import fit_regional_models"
python -c "from neuroaging.stats import compute_poststrat_weights"
```

## 📝 Common Parameters

### fit_regional_models() parameters:
- `metric_col`: Column name with brain metric values (e.g., "value", "gm_vol")
- `region_col`: Column identifying regions (default: "index")
- `age_col`: Age column name (default: "age_at_scan")
- `covariates`: List of predictors (default: ["age_at_scan", "sex"])
- `center_variables`: Center age before fitting (default: True)
- `weight_col`: Column with post-strat weights (default: "weight", None for OLS)

### configure_plotting() parameters (all optional):
- `font_family`: Font name (default: "Calibri")
- `font_path`: Path to .ttf file if needed
- `figure_size`: Tuple (width, height) in inches (default: (12, 8))
- `figure_dpi`: Display DPI (default: 200)
- `savefig_dpi`: Save DPI (default: 400)
- `style`: Seaborn style (default: "whitegrid")
- `palette`: Color palette (default: "Set2")

## 🎯 What Changes, What Stays the Same

### ✅ Same Variable Names:
- `lin_w`, `quad_w`, `w_compare` - Same as before!
- All columns in results DataFrames stay the same
- `savefig_nice()` works exactly the same way

### ✅ Same Analysis Flow:
1. Load data
2. Compute weights (if needed)
3. Fit models
4. Apply FDR correction
5. Visualize results

Only the HOW changes (importing instead of copy-pasting), not the WHAT!
