"""Source data for Supplementary Figure S5: sliding-window sensitivity grid.

Reuses the saved per-(N, step) aggregate CSVs from
notebooks/stacking_and_bag/bag_lifestyle_analysis.ipynb's parameter sweep
(N in {50,75,100,150,200}, step in {5,10,15,20}) - already-aggregate window
betas + second-stage trajectory fits, same files build_fig7.py's primary
(N=100, step=10) setting is drawn from. No rerun, no participant-level data.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from _paths import OUT_DIR, SLIDING_WINDOW_GRID_DIR

FEATURES = {
    "lateral_ventricles_mean": "lateral_ventricular_volume",
    "BMI": "BMI",
    "alcohol": "alcohol_intake",
}
N_VALUES = [50, 75, 100, 150, 200]
STEP_VALUES = [5, 10, 15, 20]
PRIMARY_N, PRIMARY_STEP = 100, 10

windows_all = []
traj_all = []
for n in N_VALUES:
    for step in STEP_VALUES:
        d = SLIDING_WINDOW_GRID_DIR / f"N{n}_S{step}"
        if not d.exists():
            print(f"WARNING: missing {d}, skipping")
            continue
        for feat, label in FEATURES.items():
            w_path = d / f"{feat}_windows.csv"
            if w_path.exists():
                w = pd.read_csv(w_path)
                w.insert(0, "phenotype", label)
                windows_all.append(w)
            for shape in ["Lin", "Quad"]:
                t_path = d / f"{feat}_trajectory_{shape}.csv"
                if not t_path.exists():
                    continue
                t = pd.read_csv(t_path)
                t.insert(0, "phenotype", label)
                t["model"] = shape.lower()
                traj_all.append(t)

figS5_windows = pd.concat(windows_all, ignore_index=True)
figS5_windows.to_csv(OUT_DIR / "figS5_sliding_window_betas_grid.csv", index=False)

figS5_traj = pd.concat(traj_all, ignore_index=True)
figS5_traj = figS5_traj.rename(columns={"mean": "beta_fit"})
figS5_traj.to_csv(OUT_DIR / "figS5_sliding_window_trajectory_grid.csv", index=False)

# ---- Point-wise Pearson correlation of each (N, step) trajectory vs the ----
# ---- primary N=100/step=10 trajectory, per phenotype/model (panel III)  ----
corr_rows = []
for feat, label in FEATURES.items():
    for shape in ["lin", "quad"]:
        primary = figS5_traj.query(
            "phenotype == @label and model == @shape and N == @PRIMARY_N and step == @PRIMARY_STEP"
        )["beta_fit"].to_numpy()
        for n in N_VALUES:
            for step in STEP_VALUES:
                alt = figS5_traj.query(
                    "phenotype == @label and model == @shape and N == @n and step == @step"
                )["beta_fit"].to_numpy()
                if len(alt) != len(primary) or len(alt) == 0:
                    continue
                r, p = pearsonr(primary, alt)
                corr_rows.append({
                    "phenotype": label, "model": shape, "N": n, "step": step,
                    "pearson_r_vs_primary": r, "pvalue": p,
                    "is_primary": (n == PRIMARY_N and step == PRIMARY_STEP),
                })
figS5_corr = pd.DataFrame(corr_rows)
figS5_corr.to_csv(OUT_DIR / "figS5c_correlation_summary.csv", index=False)

print("FigS5 done.")
print(figS5_corr.query("not is_primary").groupby(["phenotype", "model"])["pearson_r_vs_primary"].agg(["min", "mean"]))
