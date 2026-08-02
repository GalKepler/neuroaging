"""Source data for Figure 7 column II (sliding-window BAG-phenotype associations).

These CSVs already exist as saved aggregate output from
notebooks/stacking_and_bag/bag_lifestyle_analysis.ipynb (N=100, step=10 - the
paper's primary setting) - just reshape/relabel into the three panels
(ventricles, BMI, alcohol). No participant-level data touched.

Column I (global weighted model coefficients, with/without age interaction)
is NOT produced by this script: reproducing it requires
~/Projects/plasticityhub/sessions.csv and per-subject FreeSurfer stats
(/media/storage/yalab-dev/derivatives/freesurfer/...), which live outside
this repo and this machine did not have them mounted when this pipeline was
built. If you have access to those, add a global-model export (res_int /
res_raw .params and .conf_int() from plot_age_interaction() in
bag_lifestyle_analysis.ipynb) to reproduce panel I here.
"""
import pandas as pd

from _paths import BAG_WINDOWS_DIR, OUT_DIR

features = {
    "lateral_ventricles_mean": "lateral_ventricular_volume",
    "BMI": "BMI",
    "alcohol": "alcohol_intake",
}

windows_all = []
traj_all = []
for feat, label in features.items():
    w = pd.read_csv(BAG_WINDOWS_DIR / f"{feat}_windows.csv")
    w.insert(0, "phenotype", label)
    windows_all.append(w)

    for shape in ["Lin", "Quad"]:
        t = pd.read_csv(BAG_WINDOWS_DIR / f"{feat}_trajectory_{shape}.csv")
        t.insert(0, "phenotype", label)
        t["model"] = shape.lower()
        traj_all.append(t)

fig7_windows = pd.concat(windows_all, ignore_index=True)
fig7_windows.to_csv(OUT_DIR / "fig7_II_sliding_window_betas.csv", index=False)

fig7_traj = pd.concat(traj_all, ignore_index=True)
fig7_traj = fig7_traj.rename(columns={"x": "median_age", "mean": "beta_fit"})
fig7_traj.to_csv(OUT_DIR / "fig7_II_sliding_window_trajectory.csv", index=False)

print("Fig7 (column II) done. Column I (global models) NOT produced - see module docstring.")
print(fig7_windows.groupby("phenotype").size())
