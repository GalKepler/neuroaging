# Fig1 Notebooks Update Checklist

## 🎯 Goal
Update fig1 notebooks to use refactored modules. These are **low-risk** updates (visualization only).

---

## 📋 Pre-Flight Checklist

- [ ] Read `UPDATE_FIG1.md` for detailed instructions
- [ ] Read `fig1_update_example.py` output for before/after examples
- [ ] Backups ready: `cp notebooks/fig1/fig1.ipynb notebooks/fig1/fig1_backup.ipynb`
- [ ] Backups ready: `cp notebooks/fig1/fig1_redundancy.ipynb notebooks/fig1/fig1_redundancy_backup.ipynb`

---

## 🔄 Update Process

### Notebook 1: fig1.ipynb

#### Changes Needed:
- [ ] Replace matplotlib config (~100 lines → 11 lines)
- [ ] Replace compute_poststrat_weights (~130 lines → 1 line)

#### Steps:
1. [ ] Open `notebooks/fig1/fig1.ipynb` in JupyterLab/VSCode
2. [ ] Find the matplotlib config block (starts with `import matplotlib as mpl`)
3. [ ] Replace with:
   ```python
   from neuroaging.visualization import (
       configure_plotting, savefig_nice,
       COL_RAW, COL_WEIGHTED, COL_REF, COL_CENSUS,
       CMAP_RAW, CMAP_WEIGHTED
   )
   configure_plotting()
   ```
4. [ ] Find `def compute_poststrat_weights(...)` (~130 lines)
5. [ ] Replace with:
   ```python
   from neuroaging.stats import compute_poststrat_weights
   ```
6. [ ] Save notebook
7. [ ] Run "Restart Kernel and Run All Cells"
8. [ ] Verify no errors
9. [ ] Verify figures look the same

**Expected Result:** ~219 lines eliminated ✓

---

### Notebook 2: fig1_redundancy.ipynb

#### Changes Needed:
- [ ] Replace matplotlib config (~100 lines → 11 lines)

#### Steps:
1. [ ] Open `notebooks/fig1/fig1_redundancy.ipynb`
2. [ ] Find the matplotlib config block
3. [ ] Replace with the same import block as above
4. [ ] Save notebook
5. [ ] Run "Restart Kernel and Run All Cells"
6. [ ] Verify no errors
7. [ ] Verify figures look the same

**Expected Result:** ~89 lines eliminated ✓

---

## ✅ Verification

For each notebook after updating:

### Technical Verification:
- [ ] No import errors
- [ ] No runtime errors
- [ ] All cells execute successfully
- [ ] All figures generate

### Visual Verification:
- [ ] Figures have same colors
- [ ] Fonts look correct (or acceptable fallback)
- [ ] Layout is the same
- [ ] No missing plots

### Code Quality:
- [ ] No duplicated matplotlib config
- [ ] No duplicated savefig_nice function
- [ ] No duplicated compute_poststrat_weights function
- [ ] Imports are at the top of notebook

---

## 🎉 Success Criteria

You're done when:
- ✅ Both notebooks run without errors
- ✅ All figures look identical (or very similar)
- ✅ Code is cleaner and shorter
- ✅ Total ~308 lines eliminated from fig1/

---

## 📊 Expected Impact

| Notebook | Before | After | Saved |
|----------|--------|-------|-------|
| fig1.ipynb | ~230 lines config | ~11 lines | ~219 |
| fig1_redundancy.ipynb | ~100 lines config | ~11 lines | ~89 |
| **Total** | | | **~308** |

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'neuroaging'"
**Fix:** Add to first cell:
```python
import sys
sys.path.insert(0, '/home/galkepler/Projects/neuroaging')
```

### Figures look different
**Fix:** Check if you had custom settings. Pass them to `configure_plotting()`:
```python
configure_plotting(
    font_family="Calibri",
    font_path="/home/galkepler/.fonts/calibri.ttf",
    figure_size=(12, 8)
)
```

### Font warnings
**This is OK!** The font will fall back to a similar one. If you want to silence:
```python
configure_plotting(font_path="/home/galkepler/.fonts/calibri.ttf")
```

---

## 🚀 After Completing Fig1

Once fig1 notebooks work:
1. [ ] Commit changes: `git add notebooks/fig1/*.ipynb`
2. [ ] Commit message: "Refactor fig1 notebooks to use centralized modules"
3. [ ] Move to fig2 notebooks (see `REFACTORING_GUIDE.md`)

---

## 💡 Quick Reference

**Visualization import:**
```python
from neuroaging.visualization import configure_plotting, savefig_nice, COL_RAW, COL_WEIGHTED
configure_plotting()
```

**Stats import:**
```python
from neuroaging.stats import compute_poststrat_weights
```

**Test imports work:**
```bash
python -c "from neuroaging.visualization import configure_plotting; configure_plotting()"
python -c "from neuroaging.stats import compute_poststrat_weights"
```

---

## 📚 Resources

- `UPDATE_FIG1.md` - Detailed step-by-step guide
- `fig1_update_example.py` - Run to see before/after examples
- `QUICK_REFERENCE.md` - Fast lookup for common patterns
- `REFACTORING_GUIDE.md` - Complete refactoring documentation

Good luck! 🎉
