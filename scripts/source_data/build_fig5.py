"""Source data for Figure 5: co-aging networks (K=6 weighted GMM clustering).

Derived directly from the saved region_clusters.csv (already an aggregate,
454-region table produced by notebooks/regional/clustering.ipynb) - no
rerun of the GMM needed.
"""
import pandas as pd

from _paths import CLUSTERING_CSV, OUT_DIR

df = pd.read_csv(CLUSTERING_CSV)

meta_cols = [c for c in ["index", "name", "base_name", "Label Name", "network", "component", "hemisphere"] if c in df.columns]
prob_cols = [f"probability_{k}" for k in range(1, 7)]

# ---- Figure 5A: per-region cluster assignment + posterior membership probs ----
fig5a = df[meta_cols + ["cluster"] + prob_cols].copy()
fig5a.to_csv(OUT_DIR / "fig5a_cluster_membership.csv", index=False)

# ---- Figure 5B: cluster-average trajectories (GM volume & MD) + average AST ----
pred_ages = [20, 27, 35, 42, 50, 57, 65, 72, 80]
rows = []
for cluster_id, g in df.groupby("cluster"):
    for metric in ["gm_vol", "adc"]:
        row = {"cluster": cluster_id, "metric": metric, "n_regions": len(g),
               "mean_age_star": g[f"{metric}_age_star"].mean(),
               "mean_beta1": g[f"{metric}_beta1"].mean(),
               "mean_beta2": g[f"{metric}_beta2"].mean()}
        for age in pred_ages:
            row[f"pred_age_{age}"] = g[f"{metric}_pred_{age}"].mean()
        rows.append(row)
fig5b = pd.DataFrame(rows).sort_values(["cluster", "metric"])
fig5b.to_csv(OUT_DIR / "fig5b_cluster_trajectories.csv", index=False)

print("Fig5 done.")
print(df["cluster"].value_counts().sort_index())
