"""Source data for Figure 2: whole-brain age-association profiles (GM volume, MD).

Exports: (1) covariate-adjusted WLS fit curves (linear+quadratic, with 95% CI)
on an age grid, and (2) weighted binned summary stats (5-yr bins: n, mean, SEM)
as a privacy-preserving stand-in for the underlying participant scatter.
"""
import numpy as np
import pandas as pd
from statsmodels.formula.api import wls

from neuroaging.stats import compute_joint_poststrat_weights

from _paths import DATA_DIR, OUT_DIR

bad_subjects = ["IN120120"]
israel_population = pd.read_csv(DATA_DIR / "processed" / "israel_population.csv")

age_min, age_max = 15, 85
bin_width = 5

fit_rows = []
bin_rows = []

for metric, value_col in [("gm_vol", "total_gm_volume"), ("adc", "qfmean")]:
    df = pd.read_csv(DATA_DIR / "processed" / f"{metric}.csv", index_col=0).reset_index(drop=True)
    df["subject_code"] = df["subject_code"].astype(str).str.zfill(4)
    df = df.drop_duplicates(subset=["subject_code"] + (["index"] if "index" in df.columns else []), keep="last")
    df = df[~df["subject_code"].isin(bad_subjects)]
    df["sex"] = df["sex"].map({"M": 0, "F": 1})

    if metric == "gm_vol":
        d = df.drop_duplicates(subset=["subject_code"], keep="first")[
            ["age_at_scan", "sex", "tiv", value_col, "subject_code"]
        ].copy()
    else:
        d = (
            df.groupby("subject_code")
            .agg({value_col: "mean", "sex": "first", "age_at_scan": "first"})
            .reset_index()
        )

    d["weight"], _ = compute_joint_poststrat_weights(
        d, israel_population, age_col="age_at_scan", sex_col="sex", return_bin_table=True, cap=10
    )

    age_mean = d["age_at_scan"].mean()
    d["age_c"] = d["age_at_scan"] - age_mean
    covariates = ["sex", "tiv_c"] if metric == "gm_vol" else ["sex"]
    if metric == "gm_vol":
        d["tiv_c"] = d["tiv"] - d["tiv"].mean()

    lin_formula = f"{value_col} ~ age_c + " + " + ".join(covariates)
    quad_formula = lin_formula + " + I(age_c ** 2)"
    mod_lin = wls(lin_formula, data=d, weights=d["weight"]).fit()
    mod_quad = wls(quad_formula, data=d, weights=d["weight"]).fit()

    age_grid_c = np.linspace(age_min - age_mean, age_max - age_mean, 300)
    grid_df = pd.DataFrame({"age_c": age_grid_c})
    for c in covariates:
        grid_df[c] = d[c].median()

    row = {"age_years": age_grid_c + age_mean}
    for tag, model in [("linear", mod_lin), ("quadratic", mod_quad)]:
        pr = model.get_prediction(grid_df)
        ci = pr.conf_int(alpha=0.05)
        row[f"{tag}_fit"] = pr.predicted_mean
        row[f"{tag}_ci_lo"] = ci[:, 0]
        row[f"{tag}_ci_hi"] = ci[:, 1]
    fit_df = pd.DataFrame(row)
    fit_df.insert(0, "metric", metric)
    fit_rows.append((fit_df, mod_lin, mod_quad, metric))

    # Binned summary (weighted mean +/- SEM) as scatter stand-in
    bins = np.arange(age_min, age_max + bin_width, bin_width)
    d["age_bin"] = pd.cut(d["age_at_scan"], bins=bins, right=False)
    for interval, g in d.groupby("age_bin", observed=True):
        if len(g) == 0:
            continue
        w = g["weight"].to_numpy()
        vals = g[value_col].to_numpy()
        wmean = np.average(vals, weights=w)
        wvar = np.average((vals - wmean) ** 2, weights=w)
        sem = np.sqrt(wvar) / np.sqrt(len(g))
        bin_rows.append({
            "metric": metric, "age_bin_start": interval.left, "age_bin_end": interval.right,
            "n": len(g), "weighted_mean": wmean, "weighted_sem": sem,
            "unweighted_mean": vals.mean(), "unweighted_sem": vals.std(ddof=1) / np.sqrt(len(g)),
        })

fig2_fits = pd.concat([f[0] for f in fit_rows], ignore_index=True)
fig2_fits.to_csv(OUT_DIR / "fig2_fit_curves.csv", index=False)

fig2_bins = pd.DataFrame(bin_rows)
fig2_bins.to_csv(OUT_DIR / "fig2_binned_summary.csv", index=False)

model_stats = pd.DataFrame([{
    "metric": metric,
    "r2_linear": mod_lin.rsquared, "aic_linear": mod_lin.aic,
    "r2_quadratic": mod_quad.rsquared, "aic_quadratic": mod_quad.aic,
    "delta_aic": mod_quad.aic - mod_lin.aic,
    "n": int(mod_lin.nobs),
} for _, mod_lin, mod_quad, metric in fit_rows])
model_stats.to_csv(OUT_DIR / "fig2_model_stats.csv", index=False)

print("Fig2 done.")
print(model_stats)
