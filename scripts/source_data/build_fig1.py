"""Source data for Figure 1: age/sex distributions vs Israeli census.

Reproduces the aggregate quantities plotted in
notebooks/demographics_and_wholebrain/wholebrain_and_demo_viz.ipynb
(panel A: density curves; panel B: population pyramids), without exporting
any participant-level rows.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from neuroaging.stats import compute_joint_poststrat_weights
from neuroaging.visualization.demographics import get_dist

from _paths import DATA_DIR, OUT_DIR

bad_subjects = ["IN120120"]
israel_population = pd.read_csv(DATA_DIR / "processed" / "israel_population.csv")

gm = pd.read_csv(DATA_DIR / "processed" / "gm_vol.csv", index_col=0)
gm["subject_code"] = gm["subject_code"].astype(str).str.zfill(4)
gm = gm.reset_index(drop=True).drop_duplicates(subset=["subject_code", "index"], keep="last")
gm = gm[~gm["subject_code"].isin(bad_subjects)]

data_global = gm.drop_duplicates(subset=["subject_code"], keep="first")[
    ["age_at_scan", "sex", "tiv", "total_gm_volume", "strip_score", "subject_code", "session_id"]
].copy()
data_global["sex"] = data_global["sex"].map({"M": 0, "F": 1})
data_global["weight"], _ = compute_joint_poststrat_weights(
    data_global, israel_population, age_col="age_at_scan", sex_col="sex",
    return_bin_table=True, cap=10,
)

# ---- Panel A: age density curves (histplot stat="density", bin_width=2) ----
bin_width = 2
age_min, age_max = 15, 85
bins = np.arange(age_min, age_max + bin_width, bin_width)
bin_mid = (bins[:-1] + bins[1:]) / 2

# Reference/census population: simulate ages exactly as the notebook does
np.random.seed(42)
pop = israel_population.copy()
pop["age_range"] = pd.IntervalIndex.from_tuples(
    [(int(r["range_start"]), int(min(r["range_end"], 100))) for _, r in pop.iterrows()]
)
n_sim = data_global["subject_code"].nunique()
probs = pop["total"].values / pop["total"].values.sum()
counts = np.random.multinomial(n_sim, probs)
ages_sim = np.hstack([
    np.random.uniform(low=iv.left, high=iv.right, size=c)
    for iv, c in zip(pop["age_range"].to_numpy(), counts)
])


def density(ages, weights=None):
    h, _ = np.histogram(ages, bins=bins, weights=weights, density=False)
    denom = (np.sum(weights) if weights is not None else len(ages)) * bin_width
    return h / denom


fig1a = pd.DataFrame({
    "age_bin_start": bins[:-1],
    "age_bin_end": bins[1:],
    "age_bin_mid": bin_mid,
    "density_raw_snbb": density(data_global["age_at_scan"].to_numpy()),
    "density_weighted_snbb": density(data_global["age_at_scan"].to_numpy(), data_global["weight"].to_numpy()),
    "density_israel_census_ref": density(ages_sim),
})
fig1a.to_csv(OUT_DIR / "fig1a_age_density.csv", index=False)

# ---- Panel B: population pyramids (census / raw / weighted), % of total ----
pop_total = israel_population["male"].sum() + israel_population["female"].sum()
census_m = (israel_population.sort_values("range_start").set_index("range_start")["male"] / pop_total) * 100
census_f = (israel_population.sort_values("range_start").set_index("range_start")["female"] / pop_total) * 100
raw_m, raw_f = get_dist(data_global, israel_population, weight_col=None, age_col="age_at_scan", sex_col="sex")
wtd_m, wtd_f = get_dist(data_global, israel_population, weight_col="weight", age_col="age_at_scan", sex_col="sex")

fig1b = pd.DataFrame({
    "age_bin_start": census_m.index,
    "census_pct_male": census_m.values,
    "census_pct_female": census_f.values,
    "raw_sample_pct_male": raw_m.reindex(census_m.index).values,
    "raw_sample_pct_female": raw_f.reindex(census_m.index).values,
    "weighted_sample_pct_male": wtd_m.reindex(census_m.index).values,
    "weighted_sample_pct_female": wtd_f.reindex(census_m.index).values,
})
fig1b.to_csv(OUT_DIR / "fig1b_population_pyramid.csv", index=False)

# ---- Weighting summary stats (EMD, ESS etc quoted in text/Fig1 legend) ----
w = data_global["weight"].values
ess = (np.sum(w) ** 2) / np.sum(w ** 2)
weighted_age = np.average(data_global["age_at_scan"], weights=w)
raw_age = data_global["age_at_scan"].mean()
weighted_female = np.average(data_global["sex"] == 1, weights=w) * 100
raw_female = (data_global["sex"] == 1).mean() * 100
census_mid = (israel_population["range_start"] + israel_population["range_end"].replace(1000000, 95)) / 2
census_age = np.average(census_mid, weights=(israel_population["male"] + israel_population["female"]))
census_female_pct = (israel_population["female"].sum() / pop_total) * 100

# EMD, matching the notebook: binned proportions vs census bin midpoints
_pop = (
    israel_population.loc[lambda d: d["range_start"] >= 18]
    .loc[lambda d: d["range_end"] < 120]
    .sort_values("range_start")
    .reset_index(drop=True)
)
_edges = _pop["range_start"].tolist() + [_pop["range_end"].iat[-1] + 1]
_mids = (_pop["range_start"] + _pop["range_end"]) / 2
_prop_pop = (_pop["total"] / _pop["total"].sum()).to_numpy()

_bins = pd.cut(data_global["age_at_scan"], bins=_edges, right=False, include_lowest=True)
_prop_raw = _bins.value_counts(sort=False).sort_index().to_numpy(dtype=float)
_prop_raw = _prop_raw / _prop_raw.sum()
_prop_wtd = (
    data_global.groupby(_bins, observed=True)["weight"].sum()
    .reindex(pd.Categorical(_bins.cat.categories)).fillna(0.0).to_numpy(dtype=float)
)
_prop_wtd = _prop_wtd / _prop_wtd.sum()

emd_raw = wasserstein_distance(_mids, _mids, _prop_raw, _prop_pop)
emd_wtd = wasserstein_distance(_mids, _mids, _prop_wtd, _prop_pop)

summary = pd.DataFrame([{
    "n_actual": len(data_global),
    "effective_n_ess": ess,
    "max_weight": w.max(),
    "mean_age_census": census_age, "mean_age_raw": raw_age, "mean_age_weighted": weighted_age,
    "pct_female_census": census_female_pct, "pct_female_raw": raw_female, "pct_female_weighted": weighted_female,
    "EMD_years_raw_vs_census": emd_raw, "EMD_years_weighted_vs_census": emd_wtd,
}])
summary.to_csv(OUT_DIR / "fig1_weighting_summary.csv", index=False)

print("Fig1 done.", fig1a.shape, fig1b.shape, summary.shape)
