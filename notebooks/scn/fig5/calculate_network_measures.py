import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.covariance import OAS, LedoitWolf
import bct
from tqdm import tqdm


# ---------- residualize once (subjects × ROIs) ----------
def residualize_wide(
    long_df: pd.DataFrame,
    roi_col: str,
    value_col: str = "value",
    covars=("age_at_scan", "sex"),
    add_quadratic_age=True,
    add_site=False,
    site_col="site",
    add_tiv=False,
):
    d = long_df.copy()
    X = (
        d[["subject_code"] + list(set(covars))]
        .drop_duplicates("subject_code")
        .set_index("subject_code")
    )
    if add_quadratic_age and "age_at_scan" in X:
        X["age_c"] = X["age_at_scan"] - X["age_at_scan"].mean()
        X["age_c2"] = X["age_c"] ** 2
        X = X.drop(columns=["age_at_scan"])
    if add_tiv:
        assert "tiv" in long_df.columns, "TIV requested but 'tiv' not in data"
        X["tiv"] = long_df.drop_duplicates("subject_code").set_index("subject_code")["tiv"]
    if add_site and site_col in long_df.columns:
        X[site_col] = long_df.drop_duplicates("subject_code").set_index("subject_code")[site_col]
        X = pd.get_dummies(X, columns=[site_col], drop_first=True)
    if "sex" in X.columns and not np.issubdtype(X["sex"].dtype, np.number):
        X["sex"] = pd.Categorical(X["sex"]).codes
    X["const"] = 1.0
    Xv = X.values

    W = long_df.pivot(index="subject_code", columns=roi_col, values=value_col)
    W = W.loc[X.index]  # align subjects
    R = W.copy()
    for col in W.columns:
        y = W[col].to_numpy()
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            R[col] = np.nan
            continue
        beta = np.linalg.lstsq(Xv[mask], y[mask], rcond=None)[0]
        yhat = Xv @ beta
        R[col] = y - yhat
    R = R.dropna(axis=0, how="any")
    return R  # subjects × ROIs residuals


# ---------- shrinkage correlation ----------
def shrinkage_corr(X: pd.DataFrame, method="oas") -> pd.DataFrame:
    Xz = (X - X.mean(0)) / X.std(0, ddof=0).replace(0, 1)
    est = OAS() if method == "oas" else LedoitWolf()
    est.fit(Xz.values)
    C = est.covariance_
    d = np.sqrt(np.outer(np.diag(C), np.diag(C)))
    R = C / d
    np.fill_diagonal(R, 1.0)
    return pd.DataFrame(R, index=X.columns, columns=X.columns)


# ---------- map to non-negative weights & density threshold ----------
def to_positive_weights(R: pd.DataFrame, mode="abs") -> pd.DataFrame:
    V = R.values.copy()
    np.fill_diagonal(V, 0.0)
    if mode == "abs":
        V = np.abs(V)
    elif mode == "r_to_01":
        V = (V + 1.0) / 2.0
        V[V < 0] = 0.0
    else:
        raise ValueError("mode ∈ {'abs','r_to_01'}")
    return pd.DataFrame(V, index=R.index, columns=R.columns)


def threshold_density(W: pd.DataFrame, density=0.1) -> pd.DataFrame:
    A = W.values.copy()
    np.fill_diagonal(A, 0.0)
    iu = np.triu_indices_from(A, 1)
    k = int(np.floor(density * len(iu[0])))
    if k < 1:
        Z = np.zeros_like(A)
        return pd.DataFrame(Z, index=W.index, columns=W.columns)
    cutoff = np.partition(A[iu], -k)[-k]
    keep = A >= cutoff
    A_thr = np.where(keep, A, 0.0)
    np.fill_diagonal(A_thr, 0.0)
    return pd.DataFrame(A_thr, index=W.index, columns=W.columns)


# ---------- node/global metrics & AUC across densities ----------
def node_strength(W):  # abs by construction
    return pd.Series(W.values.sum(axis=1), index=W.index, name="strength")


def node_degree(W):
    return pd.Series((W.values > 0).sum(axis=1), index=W.index, name="degree")


def global_efficiency(W):
    return float(bct.efficiency_wei(W.values))


def metrics_auc(W: pd.DataFrame, densities) -> tuple[pd.Series, pd.Series, float]:
    S, D, E = [], [], []
    for d in densities:
        Wd = threshold_density(W, d)
        S.append(node_strength(Wd))
        D.append(node_degree(Wd))
        E.append(global_efficiency(Wd))
    S = pd.concat(S, axis=1).mean(1)
    D = pd.concat(D, axis=1).mean(1)
    E = float(np.mean(E))
    return S, D, E


def make_windows_by_k(age_by_subject: pd.Series, K=300, step=50, min_k=150):
    s = age_by_subject.sort_values()
    idx = s.index.to_numpy()
    windows, metas = [], []
    for start in range(0, max(1, len(idx) - min_k + 1), step):
        end = min(start + K, len(idx))
        if end - start < min_k:
            break
        ids = idx[start:end]
        ages = s.loc[ids].to_numpy()
        windows.append(ids)
        metas.append(
            {
                "median_age": float(np.median(ages)),
                "mean_age": float(np.mean(ages)),
                "sd_age": float(np.std(ages, ddof=1)),
                "n_window": int(len(ids)),
            }
        )
        if end == len(idx):
            break
    return windows, metas


def safe_win_id(median_age: float, n_window: int, prefix="win"):
    return f"{prefix}_age{median_age:05.1f}_n{n_window:03d}".replace(".", "p")


def write_window_csvs(
    long_df: pd.DataFrame,
    parcels: pd.DataFrame,
    out_dir: Path,
    *,
    roi_col="index",
    value_col="value",
    covars=("age_at_scan", "sex"),
    add_site=True,
    site_col="site",
    add_tiv=False,
    K=300,
    step=50,
    min_k=150,
    densities=np.linspace(0.05, 0.30, 6),
    shrink="oas",
    positive_mode="abs",
    force=True,
):
    """
    For each sliding age window:
      1) build SCN (shrinkage corr) on residuals
      2) convert to non-negative weights
      3) compute AUC metrics across densities
      4) write <window_id>.csv with node metrics + window metadata
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # residuals (subjects × ROIs)
    R = residualize_wide(
        long_df,
        roi_col,
        value_col,
        covars=covars,
        add_quadratic_age=True,
        add_site=add_site,
        site_col=site_col,
        add_tiv=add_tiv,
    )
    # windows on age
    subj_age = (
        long_df.drop_duplicates("subject_code")
        .set_index("subject_code")["age_at_scan"]
        .loc[R.index]
    )
    windows, metas = make_windows_by_k(subj_age, K=K, step=step, min_k=min_k)

    for ids, meta in tqdm(zip(windows, metas), total=len(metas), desc="Processing windows"):
        win_id = safe_win_id(meta["median_age"], meta["n_window"])
        target_file = out_dir / f"{win_id}.csv"
        if target_file.exists() and not force:
            continue

        Xw = R.loc[ids]
        if Xw.shape[0] < min_k:
            continue

        # SCN → positive weights
        Rw = shrinkage_corr(Xw, method=shrink)
        W = to_positive_weights(Rw, mode=positive_mode)

        # AUC over densities
        S, D, E = metrics_auc(W, densities)

        # Assemble one CSV per window
        p = parcels.copy()
        p["strength_auc"] = S.reindex(p[roi_col]).values
        p["degree_auc"] = D.reindex(p[roi_col]).values
        p["global_efficiency_auc"] = E  # scalar repeated (ok for your current format)
        p["window_id"] = win_id
        p["median_age"] = meta["median_age"]
        p["mean_age"] = meta["mean_age"]
        p["sd_age"] = meta["sd_age"]
        p["n_participants"] = meta["n_window"]

        p.to_csv(target_file, index=False)


if __name__ == "__main__":
    FORCE = True
    # metric = "adc"
    ATLAS = "schaefer2018tian2020_400_7"
    region_col = "index"
    DATA_DIR = Path("/media/storage/phd/neuroaging/data")
    metrics = ["adc", "gm_vol"]
    # Load important files

    # Load the data
    parcels = pd.read_csv(DATA_DIR / "external" / "atlases" / ATLAS / "parcels.csv", index_col=0)
    nifti = DATA_DIR / "external" / "atlases" / ATLAS / "atlas.nii.gz"
    nifti_matlab = DATA_DIR / "external" / "atlases" / ATLAS / "atlas_matlab.nii"
    distribution_metric = "qfmean"

    # Output directory for figures
    OUTPUT_DIR = Path("/media/storage/phd/neuroaging/figures/fig5") / ATLAS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bad_subjects = ["IN120120"]
    # Load the data
    data = {}
    for metric in metrics:
        data[metric] = pd.read_csv(
            DATA_DIR / "processed" / f"{metric}.csv", index_col=0
        ).reset_index(drop=True)
        # drop problematic subjects
        data[metric] = data[metric][~data[metric]["subject_code"].isin(bad_subjects)]
        data[metric]["sex"] = data[metric]["sex"].map({"M": 0, "F": 1})
    metric_cols = {
        metric: "volume" if metric == "gm_vol" else distribution_metric for metric in metrics
    }
    for m in metrics:
        df = data[m].rename(columns={metric_cols[m]: "value"}).copy()
        df = df[~df["subject_code"].isin(bad_subjects)].copy()
        df["sex"] = df["sex"].map({"M": 0, "F": 1}) if df["sex"].dtype == object else df["sex"]

        out_dir = OUTPUT_DIR / m / "network_matrices"  # one CSV per window goes here
        write_window_csvs(
            long_df=df,
            parcels=parcels,
            out_dir=out_dir,
            roi_col="index",
            value_col="value",
            covars=("age_at_scan", "sex") + (("tiv",) if m == "gm_vol" else ()),
            add_site=("site" in df.columns),
            site_col="site",
            add_tiv=("vol" in m),
            K=150,
            step=50,
            min_k=50,
            densities=np.linspace(0.05, 0.30, 6),
            shrink="oas",
            positive_mode="abs",
            force=True,
        )

    # for m, df in data.items():
    #     df = df.rename(columns={metric_cols[m]: "value"})
    #     data[m] = df

    # # Estimate SCNs
    # scns = {}
    # for m, df in data.items():
    #     scns[m] = estimate_scn(
    #         df,
    #         metric=m,
    #         region_col=region_col,
    #         age_col="age_at_scan",
    #         n_bins=50,
    #     )
    #     print(f"Estimated SCNs for {m}: {len(scns[m])} age groups")
    # # Calculate network measures
    # for m, scn in scns.items():
    #     print(f"Calculating network measures for {m}...")
    #     metrics_directory = OUTPUT_DIR / m / "network_matrices"
    #     metrics_directory.mkdir(parents=True, exist_ok=True)
    #     calculate_network_measures(
    #         scn,
    #         parcels,
    #         metrics_directory=metrics_directory,
    #         network_measures=NETWORK_MEASURES,
    #         force=FORCE,
    #     )
