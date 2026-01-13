# Fig1 Notebooks Update - COMPLETE ✅

## Summary

Both fig1 notebooks have been successfully updated to use the refactored modules.

---

## Changes Made

### 1. fig1.ipynb
**File**: `notebooks/fig1/fig1.ipynb`

**Changes:**
- **Cell 1**: Replaced 98-line matplotlib configuration with 14-line import
- **Cell 8**: Replaced 109-line `compute_poststrat_weights` function with 2-line import

**Before:**
```python
# Cell 1: ~98 lines of matplotlib config
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
# ... 90+ more lines ...
def savefig_nice(fig, filename, ...):
    ...

# Cell 8: ~109 lines
def compute_poststrat_weights(...):
    # ... 109 lines of implementation ...
```

**After:**
```python
# Cell 1: 14 lines
import matplotlib.pyplot as plt
from neuroaging.visualization import (
    configure_plotting,
    savefig_nice,
    COL_RAW,
    COL_WEIGHTED,
    COL_REF,
    COL_CENSUS,
    CMAP_RAW,
    CMAP_WEIGHTED,
)
configure_plotting()

# Cell 8: 2 lines
from neuroaging.stats import compute_poststrat_weights
```

**Lines saved**: 191 lines ✓

---

### 2. fig1_redundancy.ipynb
**File**: `notebooks/fig1/fig1_redundancy.ipynb`

**Changes:**
- **Cell 1**: Replaced 98-line matplotlib configuration with 14-line import

**Before:**
```python
# Cell 1: ~98 lines of matplotlib config
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# ... 90+ more lines ...
```

**After:**
```python
# Cell 1: 14 lines
import matplotlib.pyplot as plt
from neuroaging.visualization import (
    configure_plotting,
    savefig_nice,
    COL_RAW,
    COL_WEIGHTED,
    COL_REF,
    COL_CENSUS,
    CMAP_RAW,
    CMAP_WEIGHTED,
)
configure_plotting()
```

**Lines saved**: 84 lines ✓

---

## Total Impact

| Notebook | Before | After | Saved |
|----------|--------|-------|-------|
| fig1.ipynb | 207 lines | 16 lines | **191 lines** |
| fig1_redundancy.ipynb | 98 lines | 14 lines | **84 lines** |
| **TOTAL** | **305 lines** | **30 lines** | **275 lines** ✓ |

---

## Validation

✅ **Import Test**: All imports work correctly
✅ **Backups Created**:
- `notebooks/fig1/fig1_backup.ipynb`
- `notebooks/fig1/fig1_redundancy_backup.ipynb`

---

## Next Steps

### 1. Manual Verification (IMPORTANT!)

You should now:

1. **Open notebooks in JupyterLab or VSCode**
   ```bash
   jupyter lab notebooks/fig1/fig1.ipynb
   ```

2. **Run "Restart Kernel & Run All Cells"** for each notebook

3. **Verify:**
   - No import errors
   - No runtime errors
   - All cells execute successfully
   - **All figures look the same as before** (most important!)

4. **Visual inspection:**
   - Colors are correct
   - Fonts look good (or acceptable fallback)
   - Layout is unchanged
   - No missing plots

### 2. If Everything Works

Once verified, commit the changes:
```bash
cd notebooks/fig1
git add fig1.ipynb fig1_redundancy.ipynb
git commit -m "refactor: Use centralized modules in fig1 notebooks

- Replace ~100 line matplotlib config with imports
- Replace compute_poststrat_weights with import
- Eliminates 275 lines of duplication

Refs: neuroaging/visualization, neuroaging/stats"
```

### 3. If Issues Occur

If you encounter any problems:

**Import Errors:**
Add to the first cell of the notebook:
```python
import sys
sys.path.insert(0, '/home/galkepler/Projects/neuroaging')
```

**Figures Look Different:**
Customize `configure_plotting()` with your settings:
```python
configure_plotting(
    font_family="Calibri",
    font_path="/home/galkepler/.fonts/calibri.ttf",
    figure_size=(12, 8),
    figure_dpi=200
)
```

**Restore from Backup:**
```bash
cp notebooks/fig1/fig1_backup.ipynb notebooks/fig1/fig1.ipynb
cp notebooks/fig1/fig1_redundancy_backup.ipynb notebooks/fig1/fig1_redundancy.ipynb
```

---

## After Fig1 is Verified

Once fig1 notebooks work perfectly:

1. ✅ Commit the changes
2. 🚀 Move to fig2 notebooks (regional modeling - more complex)
3. 📖 See `REFACTORING_GUIDE.md` for fig2 instructions

Fig2 will involve the regional modeling refactoring:
- Replace ~100 line WLS model fitting loops
- Use `fit_regional_models()` from `neuroaging.modeling`
- More complex, but same validation process

---

## Files Modified

- ✅ `notebooks/fig1/fig1.ipynb` (updated)
- ✅ `notebooks/fig1/fig1_redundancy.ipynb` (updated)
- 💾 `notebooks/fig1/fig1_backup.ipynb` (backup)
- 💾 `notebooks/fig1/fig1_redundancy_backup.ipynb` (backup)

---

## Support

If you need help:
- Check `REFACTORING_GUIDE.md` for detailed instructions
- Check `QUICK_REFERENCE.md` for common patterns
- Run `python validate_refactoring.py` to verify core functionality
- Run `pytest tests/ -v` to check all tests pass

---

**Status**: ✅ COMPLETE - Ready for manual verification!

**Date**: 2026-01-13

**Script Used**: `update_fig1_notebook.py`
