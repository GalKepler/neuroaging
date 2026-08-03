"""Combine every CSV in reports/source_data/ into one Source_Data.xlsx
workbook, one sheet per figure panel, with a README cover sheet.

Run after all build_fig*.py scripts have produced their CSVs.
"""
import pandas as pd

from _paths import OUT_DIR, REPO_ROOT

SHEET_NAMES = {
    "fig1a_age_density": "Fig1a age density",
    "fig1b_population_pyramid": "Fig1b pop pyramid",
    "fig1_weighting_summary": "Fig1 weighting stats",
    "fig2_fit_curves": "Fig2 fit curves",
    "fig2_binned_summary": "Fig2 binned scatter",
    "fig2_model_stats": "Fig2 model stats",
    "fig3ab_regional_r2_fstat": "Fig3AB regional R2-F",
    "fig3c_exemplar_fit_curves": "Fig3C exemplar fits",
    "fig3c_exemplar_binned": "Fig3C exemplar binned",
    "fig4a_ast_by_region": "Fig4A AST by region",
    "fig4b_exemplar_trajectories": "Fig4B exemplar traj",
    "fig5a_cluster_membership": "Fig5A cluster membership",
    "fig5b_cluster_trajectories": "Fig5B cluster traj",
    "fig6a_model_performance": "Fig6A model performance",
    "fig6b_pred_vs_true_fit": "Fig6B pred-vs-true fit",
    "fig6c_bag_vs_age_fit": "Fig6C BAG-vs-age fit",
    "fig6bc_binned_summary": "Fig6BC binned scatter",
    "fig6bc_model_summary": "Fig6BC model summary",
    "fig6d_regional_importance": "Fig6D regional importance",
    "fig7_I_global_models": "Fig7-I global models",
    "fig7_II_sliding_window_betas": "Fig7-II window betas",
    "fig7_II_sliding_window_trajectory": "Fig7-II window traj",
    "figS1_fit_curves": "FigS1 AD-RD fit curves",
    "figS1_binned_summary": "FigS1 binned scatter",
    "figS1_model_stats": "FigS1 model stats",
    "figS2_metric_r2_matrix": "FigS2 metric R2 matrix",
    "figS2_metric_distributions": "FigS2 metric distributions",
    "figS3_fit_curves": "FigS3 WLS-vs-OLS fits",
    "figS3_model_stats": "FigS3 model stats",
    "figS4a_gmm_model_selection": "FigS4A GMM AIC-BIC sweep",
    "figS4b_cluster_topography": "FigS4B cluster topography",
    "figS4c_dice_stability": "FigS4C Dice stability",
    "figS5_sliding_window_betas_grid": "FigS5 window betas grid",
    "figS5_sliding_window_trajectory_grid": "FigS5 trajectory grid",
    "figS5c_correlation_summary": "FigS5C correlation summary",
}

README = """Source Data - Divergent Multimodal Age-Association Profiles Across the Human Brain

Every sheet contains AGGREGATED or MODEL-DERIVED results only - no
participant-level rows anywhere in this file, per data-sharing restrictions
on the underlying cohort (Helsinki approval 1356-14).

Sheets prefixed "FigS" correspond to the Supplementary Figures (S1-S5);
all other sheets correspond to the 7 main-text figures.

Panels built from participant-level scatter plots in the manuscript
(Fig 2A/B, Fig 3C, Fig 4B, Fig 6B/C) are represented here as (1) the
covariate-adjusted model fit curve with 95% CI, and (2) weighted binned
summary statistics (age bins: n, mean, SEM) as a privacy-preserving stand-in
for the raw scatter cloud.

Known gaps / discrepancies, flagged during generation:

1. Figure 3 / Figure 4 (MD quadratic-preferred count): rerunning the
   regional models against the data mounted at generation time gave
   377/454 MD regions passing the dual threshold (FDR q<0.05 AND
   deltaAIC<-15), vs the manuscript's reported 414/454. GM volume matched
   exactly (139/454). The FDR-only count for MD (414) matched the
   manuscript exactly, so the gap is ~37 regions with delta_AIC between
   about -14 and -2 - likely minor drift in the processed metric files
   since the analysis that produced the submitted numbers. Worth
   re-running against the exact data snapshot used for submission before
   this goes out as final Source Data.

Regenerate with: scripts/source_data/run_all.py (see README.md in that
folder for data-path requirements).
"""


def main():
    xlsx_path = REPO_ROOT / "reports" / "Source_Data.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame({"README": README.split("\n")}).to_excel(writer, sheet_name="README", index=False)
        for stem, sheet_name in SHEET_NAMES.items():
            csv_path = OUT_DIR / f"{stem}.csv"
            if not csv_path.exists():
                print(f"WARNING: missing {csv_path}, skipping sheet '{sheet_name}'")
                continue
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(f"Wrote {xlsx_path}")


if __name__ == "__main__":
    main()
