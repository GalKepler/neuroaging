"""Source data for Figure 6: stacked vs unimodal brain-age prediction.

Mirrors notebooks/stacking_and_bag/brain_age_prediction.ipynb (which
currently stops at a `raise` before its importance/export cells - this
script continues that pipeline standalone, without modifying the notebook).
Exports:
 (A) per-model OOF MAE +-95%CI across 5 folds, + resampled-t-test significance
 (B) predicted-vs-true age: fit line + CI + binned (5yr) weighted means
 (C) BAG vs age (post bias-correction): fit line + CI + binned weighted means
 (D) regional permutation importance (454 regions - already aggregate)
No participant-level rows are exported anywhere below.

This is the slow script in the pipeline: 7 whole-brain Ridge grid searches
plus 454 parcel-wise grid searches plus permutation importance
(n_repeats=100). Expect 20-40+ minutes depending on the machine.
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from scipy import stats
import statsmodels.api as sm

from neuroaging.utils import (
    load_metric_data,
    prep_metric_matrices,
    compute_joint_poststrat_weights,
    corrected_cross_val_predict,
    beheshti_bias_correction,
)

from _paths import DATA_DIR, OUT_DIR

ATLAS = "schaefer2018tian2020_400_7"
REGION_COL = "index"
METRICS = ["gm_vol", "wm_vol", "csf_vol", "adc", "fa", "ad", "rd"]
BAD_SUBJECTS = ["IN120120"]
COV_NAMES = {m: ["sex"] if "vol" not in m else ["sex", "tiv"] for m in METRICS}
ALPHAS = np.logspace(-3, 4, 30)
CV_SPLITS = 5
USE_WEIGHTS = False
APPLY_CORRECTION = True

parcels = pd.read_csv(DATA_DIR / "external" / "atlases" / ATLAS / "parcels.csv", index_col=0)
israel_population = pd.read_csv(DATA_DIR / "processed" / "israel_population.csv")
data = load_metric_data(DATA_DIR, METRICS, bad_subjects=BAD_SUBJECTS, distribution_metric="qfmean")
print({m: len(data[m]) for m in METRICS})

for metric in METRICS:
    data[metric]["poststrat_weight"], _ = compute_joint_poststrat_weights(
        data[metric], israel_population, age_col="age_at_scan", sex_col="sex",
        return_bin_table=True, cap=10,
    )

X_dict, y, w, cov = prep_metric_matrices(data, region_col=REGION_COL)
print(f"Common subjects: {len(y)}, age range {y.min():.1f}-{y.max():.1f}")

outer_cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=1)
pipeline = Pipeline([("scaler", RobustScaler()), ("estimator", Ridge())])
grid = GridSearchCV(pipeline, {"estimator__alpha": ALPHAS}, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)

predictions = {}
perf_rows = []

print("Fitting per-metric models...")
for metric, X in X_dict.items():
    covariates = COV_NAMES[metric]
    X_model = np.hstack([X, cov[covariates].to_numpy()])
    fit_params = {"estimator__sample_weight": w if USE_WEIGHTS else None}
    grid.fit(X_model, y, **fit_params)
    model = grid.best_estimator_

    correction_func = beheshti_bias_correction if APPLY_CORRECTION else None
    y_pred = corrected_cross_val_predict(model, X_model, y.values, cv=outer_cv, params=fit_params, correction_func=correction_func)
    y_pred_raw = cross_val_predict(model, X_model, y.values, cv=outer_cv, params=fit_params)

    predictions[metric] = pd.DataFrame({"True": y.values, "Predicted": y_pred}, index=y.index)

    fold_maes = np.array([mean_absolute_error(y.values[te], y_pred[te]) for _, te in outer_cv.split(y)])
    fold_maes_raw = np.array([mean_absolute_error(y.values[te], y_pred_raw[te]) for _, te in outer_cv.split(y)])

    perf_rows.append({
        "metric": metric, "fold_maes": fold_maes, "fold_maes_raw": fold_maes_raw,
        "R2": r2_score(y, y_pred), "MAE": mean_absolute_error(y, y_pred), "RMSE": root_mean_squared_error(y, y_pred),
    })
    print(f"{metric}: MAE={perf_rows[-1]['MAE']:.2f}, R2={perf_rows[-1]['R2']:.3f}")

print("Fitting parcel-wise base learners (this is the slow part)...")
stacked_models = parcels.copy()
base_preds = {}
for i, row in parcels.iterrows():
    X_roi = np.hstack([X_dict[m][:, [i]] for m in METRICS])
    fit_params = {"estimator__sample_weight": w if USE_WEIGHTS else None}
    grid.fit(X_roi, y, **fit_params)
    model = grid.best_estimator_
    y_pred = cross_val_predict(model, X_roi, y.values, cv=outer_cv, params=fit_params)
    base_preds[i] = y_pred
    if i % 100 == 0:
        print(f"  region {i}/{len(parcels)}")

print("Fitting meta-learner...")
X_cov = cov[COV_NAMES["gm_vol"]].to_numpy()
X_stacked = np.vstack([base_preds[i] for i in parcels.index]).T
X_stacked = np.hstack([X_stacked, X_cov])

fit_params = {"estimator__sample_weight": w if USE_WEIGHTS else None}
grid.fit(X_stacked, y, **fit_params)
meta_model = grid.best_estimator_

correction_func = beheshti_bias_correction if APPLY_CORRECTION else None
y_pred_stacked = corrected_cross_val_predict(meta_model, X_stacked, y.values, cv=outer_cv, params=fit_params, correction_func=correction_func)
y_pred_stacked_raw = cross_val_predict(meta_model, X_stacked, y.values, cv=outer_cv, params=fit_params)

predictions["stacked"] = pd.DataFrame({"True": y.values, "Predicted": y_pred_stacked}, index=y.index)

fold_maes_stacked = np.array([mean_absolute_error(y.values[te], y_pred_stacked[te]) for _, te in outer_cv.split(y)])
fold_maes_stacked_raw = np.array([mean_absolute_error(y.values[te], y_pred_stacked_raw[te]) for _, te in outer_cv.split(y)])
perf_rows.append({
    "metric": "stacked", "fold_maes": fold_maes_stacked, "fold_maes_raw": fold_maes_stacked_raw,
    "R2": r2_score(y, y_pred_stacked), "MAE": mean_absolute_error(y, y_pred_stacked), "RMSE": root_mean_squared_error(y, y_pred_stacked),
})
print(f"Stacked: MAE={perf_rows[-1]['MAE']:.2f}, R2={perf_rows[-1]['R2']:.3f}")

perf_df = pd.DataFrame(perf_rows).set_index("metric").sort_values("MAE")

# ---------------------------------------------------------------------------
# Significance: corrected resampled t-test (Nadeau & Bengio 2003) vs stacked
# ---------------------------------------------------------------------------
def corrected_resampled_ttest(mae_a, mae_b, n, k):
    d = mae_a - mae_b
    mean_d, var_d = d.mean(), d.var(ddof=1)
    n_test = n // k
    n_train = n - n_test
    corrected_var = (1 / k + n_test / n_train) * var_d
    t_stat = mean_d / np.sqrt(corrected_var)
    p_val = 2 * stats.t.sf(np.abs(t_stat), df=k - 1)
    return t_stat, p_val


n_subjects = len(y)
stacked_folds = perf_df.loc["stacked", "fold_maes_raw"]
stat_rows = []
for metric in METRICS:
    metric_folds = perf_df.loc[metric, "fold_maes_raw"]
    t, p = corrected_resampled_ttest(metric_folds, stacked_folds, n_subjects, CV_SPLITS)
    stat_rows.append({"metric": metric, "t_stat": t, "p_value": p})
stats_df = pd.DataFrame(stat_rows).set_index("metric")
stats_df["sig"] = pd.cut(stats_df["p_value"], bins=[0, 0.001, 0.01, 0.05, 1.0], labels=["***", "**", "*", "ns"])

# ---------------------------------------------------------------------------
# Fig 6A: MAE +- 95% CI per fold, with significance
# ---------------------------------------------------------------------------
fig6a_rows = []
for metric in perf_df.index:
    fm = perf_df.loc[metric, "fold_maes"]
    se = fm.std(ddof=1) / np.sqrt(len(fm))
    fig6a_rows.append({
        "metric": metric, "MAE": perf_df.loc[metric, "MAE"], "MAE_se": se,
        "MAE_ci95_lo": perf_df.loc[metric, "MAE"] - 1.96 * se, "MAE_ci95_hi": perf_df.loc[metric, "MAE"] + 1.96 * se,
        "R2": perf_df.loc[metric, "R2"], "RMSE": perf_df.loc[metric, "RMSE"],
        "n_folds": len(fm),
        "fold1_MAE": fm[0], "fold2_MAE": fm[1], "fold3_MAE": fm[2], "fold4_MAE": fm[3], "fold5_MAE": fm[4],
        "t_stat_vs_stacked": stats_df.loc[metric, "t_stat"] if metric in stats_df.index else np.nan,
        "p_value_vs_stacked": stats_df.loc[metric, "p_value"] if metric in stats_df.index else np.nan,
        "sig_vs_stacked": str(stats_df.loc[metric, "sig"]) if metric in stats_df.index else "",
    })
pd.DataFrame(fig6a_rows).to_csv(OUT_DIR / "fig6a_model_performance.csv", index=False)

# ---------------------------------------------------------------------------
# Fig 6B/C: predicted-vs-true & BAG-vs-age for the stacked model
# ---------------------------------------------------------------------------
y_true = predictions["stacked"]["True"].to_numpy()
y_pred = predictions["stacked"]["Predicted"].to_numpy()
residuals = y_pred - y_true
w_arr = w.to_numpy()

x_grid = np.linspace(y_true.min(), y_true.max(), 300)
lin_pred = sm.OLS(y_pred, sm.add_constant(y_true)).fit()
pred_summary = lin_pred.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)
fig6b_fit = pd.DataFrame({
    "true_age": x_grid, "predicted_age_fit": lin_pred.predict(sm.add_constant(x_grid)),
    "ci_lo": pred_summary["mean_ci_lower"], "ci_hi": pred_summary["mean_ci_upper"],
})
fig6b_fit.to_csv(OUT_DIR / "fig6b_pred_vs_true_fit.csv", index=False)

lin_bag = sm.OLS(residuals, sm.add_constant(y_true)).fit()
pred_summary_c = lin_bag.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)
fig6c_fit = pd.DataFrame({
    "true_age": x_grid, "BAG_fit": lin_bag.predict(sm.add_constant(x_grid)),
    "ci_lo": pred_summary_c["mean_ci_lower"], "ci_hi": pred_summary_c["mean_ci_upper"],
})
fig6c_fit.to_csv(OUT_DIR / "fig6c_bag_vs_age_fit.csv", index=False)

bin_width = 5
age_bins = np.arange(15, 90 + bin_width, bin_width)
bin_idx = pd.cut(y_true, bins=age_bins, right=False)
b_rows = []
for interval, idx in pd.Series(range(len(y_true))).groupby(bin_idx, observed=True):
    if len(idx) == 0:
        continue
    ii = idx.to_numpy()
    wt = w_arr[ii]
    pred_mean = np.average(y_pred[ii], weights=wt)
    pred_var = np.average((y_pred[ii] - pred_mean) ** 2, weights=wt)
    bag_mean = np.average(residuals[ii], weights=wt)
    bag_var = np.average((residuals[ii] - bag_mean) ** 2, weights=wt)
    b_rows.append({
        "age_bin_start": interval.left, "age_bin_end": interval.right, "n": len(ii),
        "weighted_mean_predicted_age": pred_mean, "weighted_sem_predicted_age": np.sqrt(pred_var) / np.sqrt(len(ii)),
        "weighted_mean_BAG": bag_mean, "weighted_sem_BAG": np.sqrt(bag_var) / np.sqrt(len(ii)),
    })
pd.DataFrame(b_rows).to_csv(OUT_DIR / "fig6bc_binned_summary.csv", index=False)

model_summary = pd.DataFrame([{
    "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred),
    "R2_BAG_vs_age": lin_bag.rsquared, "n": len(y_true),
}])
model_summary.to_csv(OUT_DIR / "fig6bc_model_summary.csv", index=False)

# ---------------------------------------------------------------------------
# Fig 6D: regional permutation importance (already aggregate: 454 regions)
# ---------------------------------------------------------------------------
print("Computing permutation importance (n_repeats=100)...")
perm_result = permutation_importance(
    estimator=meta_model, X=X_stacked, y=y, scoring="neg_mean_squared_error",
    n_repeats=100, random_state=42,
)
shap_imp = parcels.copy()
perm_imp = perm_result.importances_mean[:len(parcels)]
shap_imp["permutation_importance"] = perm_imp
shap_imp.loc[shap_imp["permutation_importance"] < 0, "permutation_importance"] = 0
rng = shap_imp["permutation_importance"].max() - shap_imp["permutation_importance"].min()
shap_imp["permutation_importance_scaled"] = (
    (shap_imp["permutation_importance"] - shap_imp["permutation_importance"].min()) / rng if rng > 0 else 0
)
meta_cols = [c for c in ["index", "name", "base_name", "Label Name", "network", "component", "hemisphere"] if c in shap_imp.columns]
fig6d = shap_imp[meta_cols + ["permutation_importance", "permutation_importance_scaled"]]
fig6d.to_csv(OUT_DIR / "fig6d_regional_importance.csv", index=False)

print("Fig6 done.")
print(perf_df[["MAE", "R2", "RMSE"]])
print(stats_df)
