"""Source data for Figure 7 (global + sliding-window BAG-phenotype associations).

Column II (sliding-window betas) reuses the saved aggregate CSVs from
notebooks/stacking_and_bag/bag_lifestyle_analysis.ipynb (N=100, step=10 - the
paper's primary setting) - just reshape/relabel into the three panels
(ventricles, BMI, alcohol). No participant-level data touched.

Column I (global weighted model coefficients, with/without age interaction)
re-derives res_int / res_raw from plot_age_interaction() in that notebook:
loads PlasticityHub sessions + per-subject FreeSurfer aseg stats, refits the
weighted `target ~ feature * age_centered` / `target ~ feature + age_centered`
models, and exports .params / .conf_int() plus the interaction F-test. Only
aggregate model output (coefficients, CIs, F-stats) is written - no
participant-level rows.
"""
import pandas as pd
from statsmodels.formula.api import wls
from tqdm import tqdm

from neuroaging.utils import compute_joint_poststrat_weights

from _paths import (
    BAG_PREDICTIONS_CSV,
    BAG_WINDOWS_DIR,
    DATA_DIR,
    FREESURFER_DIR,
    OUT_DIR,
    PLASTICITYHUB_SESSIONS_CSV,
)

TARGET = "stacked_BAG"

features = {
    "lateral_ventricles_mean": "lateral_ventricular_volume",
    "BMI": "BMI",
    "alcohol": "alcohol_intake",
}

# ---- Column I: global weighted models (with/without age interaction) ----

predictions_df = pd.read_csv(BAG_PREDICTIONS_CSV, index_col=0)
israel_population = pd.read_csv(DATA_DIR / "processed" / "israel_population.csv")

df_cov = pd.read_csv(PLASTICITYHUB_SESSIONS_CSV)
df_cov["subject_code"] = (
    df_cov["subject_code"].astype(str).str.replace("-", "").str.replace(" ", "").str.replace("_", "")
)
df_cov = df_cov.drop_duplicates(subset=["subject_code"]).set_index("subject_code")
df_cov["sex"] = df_cov["sex"].map({"M": 0, "F": 1})

df = df_cov.merge(predictions_df, left_on="subject_code", right_index=True, how="outer")
df = df.dropna(subset=[TARGET, "age_at_scan"])

df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
df = df.replace("", pd.NA)

for subject in tqdm(df.index, desc="Loading FreeSurfer data"):
    fs_path = FREESURFER_DIR / f"sub-{subject}" / "stats" / "aseg_stats.csv"
    if fs_path.exists():
        fs_df = pd.read_csv(fs_path, sep="\t")
        for col in fs_df.columns:
            clean_col = col.lower().replace("-", "_").replace("3rd", "third").replace("4th", "fourth")
            df.loc[subject, clean_col] = fs_df.loc[0, col]

vent_cols = [c for c in df.columns if "lateral_ventricle" in c.lower()]
df["lateral_ventricles_mean"] = df[vent_cols].mean(axis=1)

df["poststrat_weight"], _ = compute_joint_poststrat_weights(
    df, israel_population, age_col="age_at_scan", sex_col="sex", return_bin_table=True, cap=10
)
df["age_centered"] = df["age_at_scan"] - df["age_at_scan"].mean()

global_rows = []
for feat, label in features.items():
    feat_df = df.dropna(subset=[feat, TARGET, "poststrat_weight"])
    formula_int = f"{TARGET} ~ {feat} * age_centered"
    formula_raw = f"{TARGET} ~ {feat} + age_centered"
    res_int = wls(formula_int, data=feat_df, weights=feat_df["poststrat_weight"]).fit()
    res_raw = wls(formula_raw, data=feat_df, weights=feat_df["poststrat_weight"]).fit()
    f_stat, f_p, df_diff = res_int.compare_f_test(res_raw)

    for model_name, res in [("no_interaction", res_raw), ("interaction", res_int)]:
        ci = res.conf_int()
        for term in res.params.index:
            global_rows.append({
                "phenotype": label,
                "model": model_name,
                "term": term,
                "coef": res.params[term],
                "conf_int_low": ci.loc[term, 0],
                "conf_int_high": ci.loc[term, 1],
                "pvalue": res.pvalues[term],
                "n": int(res.nobs),
                "interaction_f_stat": f_stat,
                "interaction_f_pvalue": f_p,
            })

fig7_global = pd.DataFrame(global_rows)
fig7_global.to_csv(OUT_DIR / "fig7_I_global_models.csv", index=False)

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

print("Fig7 done (columns I + II).")
print(fig7_global.groupby(["phenotype", "model"]).size())
print(fig7_windows.groupby("phenotype").size())
