"""Source data for Supplementary Figure S2: global metric intercorrelation/redundancy.

Mirrors notebooks/demographics_and_wholebrain/wholebrain_redundancy.ipynb: pairwise
Pearson R^2 between whole-brain average GM volume, MD, FA, AD, RD (one value per
subject, same alignment as the notebook's scatter matrix). No participant-level
rows exported - only the resulting distribution summaries and R^2 matrix.
"""
import pandas as pd
from scipy.stats import pearsonr

from _paths import DATA_DIR, OUT_DIR

bad_subjects = ["IN120120"]
METRICS = ["gm_vol", "adc", "fa", "ad", "rd"]

data_global = {}
for metric in METRICS:
    df = pd.read_csv(DATA_DIR / "processed" / f"{metric}.csv", index_col=0).reset_index(drop=True)
    df["subject_code"] = (
        df["subject_code"].astype(str)
        .str.replace(" ", "").str.replace(".", "").str.replace("-", "").str.replace("_", "")
        .str.zfill(4)
    )
    df = df.drop_duplicates(subset=["subject_code", "index"], keep="last")
    df = df[~df["subject_code"].isin(bad_subjects)]

    if metric == "gm_vol":
        g = df.drop_duplicates(subset=["subject_code"], keep="first").rename(columns={"total_gm_volume": "qfmean"})
    else:
        g = (
            df.groupby("subject_code")
            .agg({"qfmean": "mean", "sex": "first", "age_at_scan": "first"})
            .reset_index()
        )
    data_global[metric] = g[["subject_code", "qfmean", "age_at_scan"]]

# ---- Pairwise R^2 matrix (upper triangle of Fig S2) ----
r2_rows = []
for i, m1 in enumerate(METRICS):
    for m2 in METRICS[i + 1:]:
        merged = pd.merge(data_global[m1], data_global[m2], on="subject_code", suffixes=("_x", "_y"))
        r, p = pearsonr(merged["qfmean_x"], merged["qfmean_y"])
        r2_rows.append({"metric_1": m1, "metric_2": m2, "r": r, "r2": r ** 2, "pvalue": p, "n": len(merged)})
fig_s2_r2 = pd.DataFrame(r2_rows)
fig_s2_r2.to_csv(OUT_DIR / "figS2_metric_r2_matrix.csv", index=False)

# ---- Per-metric distribution summary (diagonal panels) ----
dist_rows = []
for metric in METRICS:
    g = data_global[metric]
    dist_rows.append({
        "metric": metric, "n": len(g),
        "mean": g["qfmean"].mean(), "std": g["qfmean"].std(),
        "min": g["qfmean"].min(), "max": g["qfmean"].max(),
        "median": g["qfmean"].median(),
    })
pd.DataFrame(dist_rows).to_csv(OUT_DIR / "figS2_metric_distributions.csv", index=False)

print("FigS2 done.")
print(fig_s2_r2)
