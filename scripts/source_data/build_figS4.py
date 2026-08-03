"""Source data for Supplementary Figure S4: co-aging clustering sensitivity.

Mirrors notebooks/regional/clustering.ipynb's weighted-vs-unweighted GMM
comparison. Panel A refits the K=2..20 model-selection sweep (AIC/BIC) for
both the R2-weighted and unweighted GMM, since that curve isn't cached
anywhere on disk. Panels B/C (spatial topography + Dice stability) reuse the
saved, already-fitted cluster assignments in
figures/revision/fig2_clustering/{weighted,unweighted}/region_clusters.csv
(same files build_fig5.py's CLUSTERING_CSV points at for the weighted side)
rather than refitting the final model, to stay consistent with the cached
manuscript result. Matched-K (K=5, K=6) labels for the Dice comparison come
from the sweep itself (each K's fitted labels are kept in memory).

SLOW: like build_fig6.py, this fits a GaussianMixture at every K in 2..20
twice (weighted + unweighted) with n_init=1000, mirroring the notebook's
settings exactly - budget 30-60+ minutes.
"""
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from neuroaging.stats import compute_joint_poststrat_weights
from neuroaging.modeling.regional import fit_regional_models, apply_fdr_correction
from neuroaging.modeling.ast import stabilization_age

from _paths import CLUSTERING_CSV, DATA_DIR, OUT_DIR, UNWEIGHTED_CLUSTERING_CSV

ATLAS = "schaefer2018tian2020_400_7"
region_col = "index"
K_RANGE = range(2, 20)

WEIGHTED_CLUSTERS_CSV = CLUSTERING_CSV
UNWEIGHTED_CLUSTERS_CSV = UNWEIGHTED_CLUSTERING_CSV


class WeightedGaussianMixture(GaussianMixture):
    """Same custom weighted-EM GMM used in notebooks/regional/clustering.ipynb."""

    def fit(self, X, y=None, sample_weight=None):
        self.sample_weight = sample_weight
        return super().fit(X, y)

    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        resp = np.exp(log_resp)
        w_resp = resp * self.sample_weight[:, np.newaxis] if self.sample_weight is not None else resp

        nk = w_resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / nk.sum()
        self.means_ = np.dot(w_resp.T, X) / nk[:, np.newaxis]
        self.covariances_ = self._estimate_weighted_covariances(X, w_resp, nk, self.means_, self.covariance_type)

        if self.covariance_type == "full":
            for k in range(self.n_components):
                self.covariances_[k].flat[:: X.shape[1] + 1] += self.reg_covar
        elif self.covariance_type == "diag":
            self.covariances_ += self.reg_covar

        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)

    def _estimate_weighted_covariances(self, X, w_resp, nk, means, covariance_type):
        n_components, n_features = means.shape
        if covariance_type == "full":
            covariances = np.empty((n_components, n_features, n_features))
            for k in range(n_components):
                diff = X - means[k]
                covariances[k] = np.dot(w_resp[:, k] * diff.T, diff) / nk[k]
            return covariances
        elif covariance_type == "diag":
            covariances = np.empty((n_components, n_features))
            for k in range(n_components):
                diff = X - means[k]
                covariances[k] = np.dot(w_resp[:, k], diff ** 2) / nk[k]
            return covariances
        raise NotImplementedError(f"Covariance type {covariance_type} not implemented.")


def dice(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    return (2 * len(set_a & set_b)) / (len(set_a) + len(set_b))


def greedy_dice_alignment(labels_a, labels_b):
    """Greedy-match clusters of labels_a (source) onto labels_b (reference) by Dice."""
    a_ids, b_ids = sorted(set(labels_a)), sorted(set(labels_b))
    sets_a = {a: set(np.flatnonzero(labels_a == a)) for a in a_ids}
    sets_b = {b: set(np.flatnonzero(labels_b == b)) for b in b_ids}
    pairs = sorted(
        [(a, b, dice(sets_a[a], sets_b[b])) for a in a_ids for b in b_ids],
        key=lambda t: t[2], reverse=True,
    )
    used_a, used_b, matched = set(), set(), []
    for a, b, d in pairs:
        if a in used_a or b in used_b:
            continue
        matched.append({"reference_cluster": b, "matched_cluster": a, "dice": d})
        used_a.add(a)
        used_b.add(b)
    return pd.DataFrame(matched)


# ---------------------------------------------------------------------------
# Rebuild the region-level clustering features (same recipe as clustering.ipynb)
# ---------------------------------------------------------------------------
parcels = pd.read_csv(DATA_DIR / "external" / "atlases" / ATLAS / "parcels.csv", index_col=0)
israel_population = pd.read_csv(DATA_DIR / "processed" / "israel_population.csv")
bad_subjects = ["IN120120"]

metrics = ["adc", "gm_vol"]
distribution_metric = "qfmean"

all_data = {}
for metric in metrics:
    data = pd.read_csv(DATA_DIR / "processed" / f"{metric}.csv", index_col=0).reset_index(drop=True)
    data = data[~data["subject_code"].isin(bad_subjects)]
    data["sex"] = data["sex"].map({"M": 0, "F": 1})
    data["weight"], _ = compute_joint_poststrat_weights(
        data, israel_population, age_col="age_at_scan", sex_col="sex", return_bin_table=True, cap=10,
    )
    all_data[metric] = data

results = {}
for metric, data in all_data.items():
    metric_col = "volume" if metric == "gm_vol" else distribution_metric
    covariates = ["age_c", "C(sex)"] + (["tiv"] if metric == "gm_vol" else [])
    print(f"Fitting regional models for {metric} ...")
    models = fit_regional_models(
        data, parcels, region_col=region_col, metric_col=metric_col,
        covariates=covariates, center_variables=True,
    )
    apply_fdr_correction(models)
    results[metric] = models

age_grid = np.linspace(20, 80, 9)
features_df = parcels.copy()
coeff_features = []

for metric in metrics:
    lin, quad, compare = results[metric].linear, results[metric].quadratic, results[metric].comparison
    mean_age = all_data[metric]["age_at_scan"].mean()
    best = pd.DataFrame(index=parcels.index)
    for i in parcels.index:
        delta_aic = compare.loc[i, "delta_AIC"]
        p_value = compare.loc[i, "pvalue"]
        use_quad = (delta_aic < -15) and (p_value < (0.05 / len(parcels)))
        model = quad.loc[i, "model"] if use_quad else lin.loc[i, "model"]
        r2_adj = quad.loc[i, "r2_adj"] if use_quad else lin.loc[i, "r2_adj"]
        best.loc[i, "intercept"] = model.params["Intercept"]
        best.loc[i, "beta1"] = model.params["age_c"]
        best.loc[i, "se_beta1"] = model.bse["age_c"]
        best.loc[i, "r2_adj"] = r2_adj
        if use_quad:
            best.loc[i, "beta2"] = model.params["I(age_c ** 2)"]
            best.loc[i, "se_beta2"] = model.bse["I(age_c ** 2)"]
            age_star, _, _ = stabilization_age(
                metric, {"age": model.params["age_c"], "age_sq": model.params["I(age_c ** 2)"]}, mean_age,
            )
        else:
            best.loc[i, "beta2"] = 0
            best.loc[i, "se_beta2"] = 0
            age_star = 18
        best.loc[i, "age_star"] = age_star
    results[metric].best_w = best

    for feature in ["beta1", "beta2", "age_star"]:
        feat_name = f"{metric}_{feature}"
        features_df[feat_name] = best[feature]
        coeff_features.append(feat_name)

eps = 1e-6
weights = []
for i in parcels.index:
    metric_weights = []
    for metric in metrics:
        best = results[metric].best_w
        r2 = max(best.loc[i, "r2_adj"], 0)
        se1 = best.loc[i, "se_beta1"] if np.isfinite(best.loc[i, "se_beta1"]) else 0.0
        se2 = best.loc[i, "se_beta2"] if np.isfinite(best.loc[i, "se_beta2"]) else 0.0
        metric_weights.append(r2 * (1.0 / (se1 + se2 + eps)))
    weights.append(np.mean(metric_weights))
weights = np.clip(np.asarray(weights, dtype=float), *np.nanpercentile(weights, [5, 95]))
weights = np.clip(weights, 1e-6, None)
weights = weights / weights.sum()
weights_series = pd.Series(weights, index=parcels.index)

X = features_df[coeff_features].astype(float).replace([np.inf, -np.inf], np.nan)
valid_mask = np.isfinite(X).all(axis=1)
X = X.loc[valid_mask]
weights_series = weights_series.loc[X.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ---------------------------------------------------------------------------
# Panel A: AIC/BIC sweep, K=2..20, weighted vs unweighted
# ---------------------------------------------------------------------------
sweep_rows = []
matched_labels = {}  # (use_weights, k) -> label array, for the matched-K Dice comparison

for use_weights in [True, False]:
    tag = "weighted" if use_weights else "unweighted"
    sw = weights_series.values if use_weights else None
    for k in tqdm(K_RANGE, desc=f"GMM sweep ({tag})", unit="K"):
        gmm = WeightedGaussianMixture(
            n_components=k, random_state=0, n_init=1000, max_iter=1000,
            init_params="random", reg_covar=1e-4, covariance_type="full",
        )
        gmm.fit(X_scaled_df, sample_weight=sw)
        labels = gmm.predict(X_scaled)
        sil = silhouette_score(X_scaled_df, labels) if k > 1 else np.nan
        sweep_rows.append({
            "weighting": tag, "k": k,
            "aic": gmm.aic(X_scaled_df), "bic": gmm.bic(X_scaled_df), "silhouette": sil,
        })
        if k in (5, 6):
            matched_labels[(tag, k)] = labels

sweep_df = pd.DataFrame(sweep_rows)
for tag, g in sweep_df.groupby("weighting"):
    g = g.set_index("k")
    best_k_aic = KneeLocator(g.index, g["aic"].astype(float), curve="convex", direction="decreasing", S=1).knee
    best_k_bic = KneeLocator(g.index, g["bic"].astype(float), curve="convex", direction="decreasing", S=1).knee
    sweep_df.loc[sweep_df["weighting"] == tag, "best_k_aic_knee"] = best_k_aic
    sweep_df.loc[sweep_df["weighting"] == tag, "best_k_bic_knee"] = best_k_bic
sweep_df.to_csv(OUT_DIR / "figS4a_gmm_model_selection.csv", index=False)

# ---------------------------------------------------------------------------
# Panel B: spatial cluster assignment - reuse the cached final-model labels
# (weighted K=6, unweighted K=5/6 as saved) rather than refitting.
# ---------------------------------------------------------------------------
meta_cols = [c for c in ["index", "name", "base_name", "Label Name", "network", "component", "hemisphere"] if c in parcels.columns]
topo_rows = []
for tag, path in [("weighted", WEIGHTED_CLUSTERS_CSV), ("unweighted", UNWEIGHTED_CLUSTERS_CSV)]:
    if not path.exists():
        print(f"WARNING: missing cached cluster assignment for {tag} at {path}, skipping panel B/C for it.")
        continue
    df = pd.read_csv(path)
    cols = [c for c in meta_cols if c in df.columns] + ["cluster"]
    t = df[cols].copy()
    t.insert(0, "weighting", tag)
    topo_rows.append(t)
if topo_rows:
    pd.concat(topo_rows, ignore_index=True).to_csv(OUT_DIR / "figS4b_cluster_topography.csv", index=False)

# ---------------------------------------------------------------------------
# Panel C: Dice stability - primary weighted vs unweighted (cached labels),
# plus matched-K (K=5, K=6) comparisons from the sweep above.
# ---------------------------------------------------------------------------
dice_rows = []
if topo_rows:
    w_df = pd.read_csv(WEIGHTED_CLUSTERS_CSV)
    u_df = pd.read_csv(UNWEIGHTED_CLUSTERS_CSV)
    merged = pd.merge(w_df[["index", "cluster"]], u_df[["index", "cluster"]], on="index", suffixes=("_weighted", "_unweighted"))
    align = greedy_dice_alignment(merged["cluster_unweighted"].to_numpy(), merged["cluster_weighted"].to_numpy())
    align.insert(0, "comparison", "primary_weighted_vs_unweighted_cached")
    dice_rows.append(align)

for k in (5, 6):
    if ("weighted", k) in matched_labels and ("unweighted", k) in matched_labels:
        align = greedy_dice_alignment(matched_labels[("unweighted", k)], matched_labels[("weighted", k)])
        align.insert(0, "comparison", f"matched_K{k}")
        dice_rows.append(align)

if dice_rows:
    dice_df = pd.concat(dice_rows, ignore_index=True)
    dice_df.to_csv(OUT_DIR / "figS4c_dice_stability.csv", index=False)
    print(dice_df.groupby("comparison")["dice"].agg(["mean", "min", "max"]))

print("FigS4 done.")
