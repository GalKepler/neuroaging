"""Source data for Figure 3 (regional R2/F-stat maps + exemplar scatters) and
Figure 4 (AST maps + exemplar trajectories). Both reuse the same 454-region
linear/quadratic WLS fits, so they're computed together.

NOTE: rerunning this against a different snapshot of adc.csv/gm_vol.csv than
the one used for the manuscript can shift borderline regions across the
dual-threshold (FDR q<0.05 AND deltaAIC<-15) cutoff. When this was run
against the Minerva-mounted data, GM volume matched the manuscript exactly
(139/454 quadratic-preferred) but MD came out at 377/454 vs the manuscript's
414/454 - the FDR-only count for MD (414) matched exactly, so the gap is in
borderline deltaAIC values (~-14 to -2) for ~37 regions. Worth re-checking
against whatever data snapshot the submission was generated from.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from neuroaging.stats import compute_joint_poststrat_weights
from neuroaging.modeling.regional import fit_regional_models, apply_fdr_correction
from neuroaging.modeling.ast import stabilization_age

from _paths import DATA_DIR, OUT_DIR

ATLAS = "schaefer2018tian2020_400_7"
region_col = "index"

parcels = pd.read_csv(DATA_DIR / "external" / "atlases" / ATLAS / "parcels.csv", index_col=0)
israel_population = pd.read_csv(DATA_DIR / "processed" / "israel_population.csv")
bad_subjects = ["IN120120"]

meta_cols = [c for c in ["index", "name", "base_name", "Label Name", "network", "component", "hemisphere"] if c in parcels.columns]

metrics = ["gm_vol", "adc"]
distribution_metric = "qfmean"
metric_cols = {"gm_vol": "volume", "adc": distribution_metric}

results = {}
all_data = {}
for metric in metrics:
    data = pd.read_csv(DATA_DIR / "processed" / f"{metric}.csv", index_col=0).reset_index(drop=True)
    data = data[~data["subject_code"].isin(bad_subjects)]
    data["sex"] = data["sex"].map({"M": 0, "F": 1})
    data["weight"], _ = compute_joint_poststrat_weights(
        data, israel_population, age_col="age_at_scan", sex_col="sex", return_bin_table=True, cap=10,
    )
    all_data[metric] = data

    covariates = ["age_c", "sex"] + (["tiv"] if metric == "gm_vol" else [])
    print(f"Fitting regional models for {metric} ...")
    models = fit_regional_models(
        data, parcels, region_col=region_col, metric_col=metric_cols[metric],
        covariates=covariates, center_variables=True,
    )
    apply_fdr_correction(models)
    results[metric] = models
    print(f"  done: {metric}")

# ---------------------------------------------------------------------------
# Figure 3A/3B: regional R2 (quadratic) and partial-F (log10) maps
# ---------------------------------------------------------------------------
fig3ab_rows = []
for metric, models in results.items():
    q = models.quadratic
    comp = models.comparison
    for i, row in parcels.iterrows():
        fig3ab_rows.append({
            **{c: row[c] for c in meta_cols},
            "metric": metric,
            "r2_adj_quadratic": q.loc[i, "r2_adj"],
            "r2_linear": models.linear.loc[i, "r2"],
            "log10_partial_F": np.log10(comp.loc[i, "f"]) if comp.loc[i, "f"] > 0 else np.nan,
            "partial_F_pvalue_fdr": comp.loc[i, "pvalue_corrected"],
            "delta_AIC": comp.loc[i, "delta_AIC"],
            "quadratic_preferred": bool((comp.loc[i, "pvalue_corrected"] < 0.05) and (comp.loc[i, "delta_AIC"] < -15)),
        })
fig3ab = pd.DataFrame(fig3ab_rows)
fig3ab.to_csv(OUT_DIR / "fig3ab_regional_r2_fstat.csv", index=False)
print("n quadratic-preferred (gm_vol):", fig3ab.query("metric=='gm_vol'")["quadratic_preferred"].sum())
print("n quadratic-preferred (adc):", fig3ab.query("metric=='adc'")["quadratic_preferred"].sum())

# ---------------------------------------------------------------------------
# Figure 3C: exemplar region fit curves + binned scatter stand-in
# gm_vol rois=[48,405] (most linear vs most quadratic); adc rois=[50,234]
# ---------------------------------------------------------------------------
exemplar_rois = {"gm_vol": [48, 405], "adc": [50, 234]}
age_min, age_max = 15, 90
bin_width = 5

fig3c_fit_rows = []
fig3c_bin_rows = []
for metric, rois in exemplar_rois.items():
    data = all_data[metric]
    age_mean = data["age_at_scan"].mean()
    for i in rois:
        lm = results[metric].linear.loc[i, "model"]
        qm = results[metric].quadratic.loc[i, "model"]
        region_label = parcels.loc[i, "name"] if "name" in parcels.columns else str(i)

        d = data[data[region_col] == parcels.loc[i, region_col]].copy()
        value_col = metric_cols[metric]

        age_grid_c = np.linspace(age_min - age_mean, age_max - age_mean, 300)
        grid_df = pd.DataFrame({"age_c": age_grid_c})
        covs = [v for v in qm.model.exog_names if v not in ["Intercept", "age_c", "I(age_c ** 2)"]]
        for c in covs:
            grid_df[c] = d[c].median() if pd.api.types.is_numeric_dtype(d[c]) else d[c].mode()[0]

        for tag, model in [("linear", lm), ("quadratic", qm)]:
            pr = model.get_prediction(grid_df)
            ci = pr.conf_int(alpha=0.05)
            for age_c, fit, lo, hi in zip(age_grid_c, pr.predicted_mean, ci[:, 0], ci[:, 1]):
                fig3c_fit_rows.append({
                    "metric": metric, "region_index": i, "region_name": region_label,
                    "model": tag, "age_years": age_c + age_mean, "fit": fit, "ci_lo": lo, "ci_hi": hi,
                })

        bins = np.arange(age_min, age_max + bin_width, bin_width)
        d["age_bin"] = pd.cut(d["age_at_scan"], bins=bins, right=False)
        for interval, g in d.groupby("age_bin", observed=True):
            if len(g) == 0:
                continue
            w = g["weight"].to_numpy()
            vals = g[value_col].to_numpy()
            wmean = np.average(vals, weights=w)
            wvar = np.average((vals - wmean) ** 2, weights=w)
            fig3c_bin_rows.append({
                "metric": metric, "region_index": i, "region_name": region_label,
                "age_bin_start": interval.left, "age_bin_end": interval.right,
                "n": len(g), "weighted_mean": wmean, "weighted_sem": np.sqrt(wvar) / np.sqrt(len(g)),
            })

pd.DataFrame(fig3c_fit_rows).to_csv(OUT_DIR / "fig3c_exemplar_fit_curves.csv", index=False)
pd.DataFrame(fig3c_bin_rows).to_csv(OUT_DIR / "fig3c_exemplar_binned.csv", index=False)

# ---------------------------------------------------------------------------
# Figure 4A: AST per region per metric
# ---------------------------------------------------------------------------
fig4a_rows = []
for metric, data in all_data.items():
    mean_age = data["age_at_scan"].mean()
    for i, row in tqdm(parcels.iterrows(), total=parcels.shape[0], desc=f"AST {metric}"):
        quad_model = results[metric].quadratic.loc[i, "model"]
        delta_aic = results[metric].comparison.loc[i, "delta_AIC"]
        beta1 = quad_model.params["age_c"]
        beta2 = quad_model.params["I(age_c ** 2)"]
        if delta_aic < -15:
            age_star, clipped, u = stabilization_age(metric, {"age": beta1, "age_sq": beta2}, mean_age)
        else:
            age_star, clipped, u = 18, True, (beta2 > 0)
        fig4a_rows.append({
            **{c: row[c] for c in meta_cols},
            "metric": metric, "beta1": beta1, "beta2": beta2,
            "age_star": age_star, "clipped": clipped, "u_shaped": u,
            "delta_AIC": delta_aic,
        })
fig4a = pd.DataFrame(fig4a_rows)
fig4a.to_csv(OUT_DIR / "fig4a_ast_by_region.csv", index=False)

# ---------------------------------------------------------------------------
# Figure 4B: exemplar trajectories - hippocampus(405), lateral PFC(179), sensory(50)
# ---------------------------------------------------------------------------
example_rois = [405, 179, 50]
fig4b_fit_rows = []
for i in example_rois:
    region_label = parcels.loc[i, "name"] if "name" in parcels.columns else str(i)
    for metric in metrics:
        data = all_data[metric]
        mean_age = data["age_at_scan"].mean()
        lm = results[metric].linear.loc[i, "model"]
        qm = results[metric].quadratic.loc[i, "model"]
        d = data[data[region_col] == parcels.loc[i, region_col]].copy()

        age_grid_c = np.linspace(age_min - mean_age, age_max - mean_age, 200)
        grid_df = pd.DataFrame({"age_c": age_grid_c})
        covs = [v for v in qm.model.exog_names if v not in ["Intercept", "age_c", "I(age_c ** 2)"]]
        for c in covs:
            grid_df[c] = d[c].median() if pd.api.types.is_numeric_dtype(d[c]) else d[c].mode()[0]

        ast_row = fig4a[(fig4a["index"] == parcels.loc[i, "index"]) & (fig4a["metric"] == metric)]
        age_star = float(ast_row["age_star"].iloc[0]) if len(ast_row) else np.nan

        for tag, model in [("linear", lm), ("quadratic", qm)]:
            pr = model.get_prediction(grid_df)
            ci = pr.conf_int(alpha=0.05)
            for age_c, fit, lo, hi in zip(age_grid_c, pr.predicted_mean, ci[:, 0], ci[:, 1]):
                fig4b_fit_rows.append({
                    "region_index": i, "region_name": region_label, "metric": metric,
                    "model": tag, "age_years": age_c + mean_age, "fit": fit, "ci_lo": lo, "ci_hi": hi,
                    "AST": age_star,
                })

pd.DataFrame(fig4b_fit_rows).to_csv(OUT_DIR / "fig4b_exemplar_trajectories.csv", index=False)

print("Fig3/Fig4 done.")
