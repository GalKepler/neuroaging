"""Source data for Supplementary Figure S3: weighting sensitivity (WLS vs OLS).

Mirrors the USE_WEIGHTS=False comparison block of
notebooks/demographics_and_wholebrain/wholebrain_and_demo_viz.ipynb: quadratic
OLS (unweighted) vs quadratic WLS (population-calibrated) fits of whole-brain
GM volume and MD on age, adjusted for sex (+ TIV for GM volume).
"""
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols, wls

from neuroaging.stats import compute_joint_poststrat_weights

from _paths import DATA_DIR, OUT_DIR

bad_subjects = ["IN120120"]
israel_population = pd.read_csv(DATA_DIR / "processed" / "israel_population.csv")

age_min, age_max = 15, 85

fit_rows = []
model_rows = []

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

    quad_formula = f"{value_col} ~ age_c + I(age_c ** 2) + " + " + ".join(covariates)
    mod_unw = ols(quad_formula, data=d).fit()
    mod_wtd = wls(quad_formula, data=d, weights=d["weight"]).fit()

    age_grid_c = np.linspace(age_min - age_mean, age_max - age_mean, 300)
    grid_df = pd.DataFrame({"age_c": age_grid_c})
    for c in covariates:
        grid_df[c] = d[c].median()

    row = {"age_years": age_grid_c + age_mean}
    for tag, model in [("unweighted_ols", mod_unw), ("weighted_wls", mod_wtd)]:
        pr = model.get_prediction(grid_df)
        ci = pr.conf_int(alpha=0.05)
        row[f"{tag}_fit"] = pr.predicted_mean
        row[f"{tag}_ci_lo"] = ci[:, 0]
        row[f"{tag}_ci_hi"] = ci[:, 1]
    fit_df = pd.DataFrame(row)
    fit_df.insert(0, "metric", metric)
    fit_rows.append(fit_df)

    model_rows.append({
        "metric": metric,
        "r2_unweighted": mod_unw.rsquared, "aic_unweighted": mod_unw.aic,
        "r2_weighted": mod_wtd.rsquared, "aic_weighted": mod_wtd.aic,
        "n": int(mod_unw.nobs),
    })

pd.concat(fit_rows, ignore_index=True).to_csv(OUT_DIR / "figS3_fit_curves.csv", index=False)
model_stats = pd.DataFrame(model_rows)
model_stats.to_csv(OUT_DIR / "figS3_model_stats.csv", index=False)

print("FigS3 done.")
print(model_stats)
