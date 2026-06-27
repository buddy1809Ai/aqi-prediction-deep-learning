
"""
BLOCK 45 — GITHUB EXPORT MAP
Generates outputs/github_export_map.md — complete mapping table:
  Repository File | Source Block | Status
Covers every file in the GitHub repo. No training.
"""
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SEP = "=" * 70

# ── Complete file manifest ─────────────────────────────────────────────────
# Format: (repo_path, source_block, status, notes)
MANIFEST = [
    # ── Root
    ("README.md",                              "42_GitHub_Readiness",          "READY",   "Generated template — expand before push"),
    ("requirements.txt",                       "42_GitHub_Readiness",          "READY",   "10 packages pinned"),
    (".gitignore",                             "42_GitHub_Readiness",          "READY",   "Excludes raw data, models, __pycache__"),

    # ── outputs/cleaned (parquets — may be too large for GitHub; use Git LFS or exclude)
    ("outputs/cleaned/Ahmedabad_cleaned.parquet",       "04_Data_Cleaning", "GIT-LFS", "1.6 MB — use Git LFS or data/sample only"),
    ("outputs/cleaned/Chennai_cleaned.parquet",         "04_Data_Cleaning", "GIT-LFS", "2.4 MB"),
    ("outputs/cleaned/Delhi_NCR_cleaned.parquet",       "04_Data_Cleaning", "GIT-LFS", "3.0 MB"),
    ("outputs/cleaned/GandhiNagar_cleaned.parquet",     "04_Data_Cleaning", "GIT-LFS", "3.1 MB"),
    ("outputs/cleaned/Hyderabad_cleaned.parquet",       "04_Data_Cleaning", "GIT-LFS", "4.9 MB"),
    ("outputs/cleaned/Indore_cleaned.parquet",          "04_Data_Cleaning", "GIT-LFS", "1.5 MB"),
    ("outputs/cleaned/Jaipur_cleaned.parquet",          "04_Data_Cleaning", "GIT-LFS", "2.8 MB"),
    ("outputs/cleaned/Jodhpur_cleaned.parquet",         "04_Data_Cleaning", "GIT-LFS", "2.7 MB"),
    ("outputs/cleaned/Mumbai_cleaned.parquet",          "04_Data_Cleaning", "GIT-LFS", "3.0 MB"),
    ("outputs/cleaned/Mumbai_suburbs_cleaned.parquet",  "04_Data_Cleaning", "GIT-LFS", "1.5 MB"),
    ("outputs/cleaned/Nagpur_cleaned.parquet",          "04_Data_Cleaning", "GIT-LFS", "4.5 MB"),
    ("outputs/cleaned/Pune_cleaned.parquet",            "04_Data_Cleaning", "GIT-LFS", "3.4 MB"),
    ("outputs/cleaned/Singrauli_cleaned.parquet",       "04_Data_Cleaning", "GIT-LFS", "1.8 MB"),
    ("outputs/cleaned/Surat_cleaned.parquet",           "04_Data_Cleaning", "GIT-LFS", "1.4 MB"),
    ("outputs/cleaned/Thane_cleaned.parquet",           "04_Data_Cleaning", "GIT-LFS", "1.4 MB"),
    ("outputs/cleaned/Vapi_cleaned.parquet",            "04_Data_Cleaning", "GIT-LFS", "2.2 MB"),
    ("outputs/cleaned/bhopal_cleaned.parquet",          "04_Data_Cleaning", "GIT-LFS", "1.5 MB"),
    ("outputs/cleaned/vishakhapattanam_cleaned.parquet","04_Data_Cleaning", "GIT-LFS", "1.4 MB"),

    # ── outputs/recovered (parquets)
    ("outputs/recovered/[18 city files]",      "P1_City_Recovery",             "GIT-LFS", "~5–21 MB each — use Git LFS"),

    # ── Track A results CSVs
    ("outputs/track_a_ridge.csv",              "17_Track_A_Ridge_All_Cities",  "READY",   "18 rows"),
    ("outputs/track_a_rf.csv",                 "18_Track_A_RF_All_Cities",     "READY",   "18 rows"),
    ("outputs/track_a_gbr.csv",                "19_Track_A_GBR_All_Cities",    "READY",   "18 rows"),
    ("outputs/track_a_xgb.csv",                "20_Track_A_XGB_All_Cities",    "READY",   "18 rows"),
    ("outputs/final_track_a_lstm.csv",         "16_Track_A_LSTM_All_Cities",   "READY",   "18 rows"),
    ("outputs/track_a_bilstm.csv",             "30_Track_A_BiLSTM",            "READY",   "18 rows"),
    ("outputs/track_a_cnn_bilstm.csv",         "31_Track_A_CNN_BiLSTM",        "READY",   "18 rows"),
    ("outputs/final_track_a_complete.csv",     "28_Scientific_Comparison",     "READY",   "126 rows — master Track A table"),
    ("outputs/final_track_a_classical.csv",    "28_Scientific_Comparison",     "READY",   "Classical subset"),
    ("outputs/track_a_estimation_results.csv", "13_Track_A_Estimation",        "REVIEW",  "Pilot run (3 cities) — superseded by complete"),

    # ── Track B results CSVs
    ("outputs/track_b_rf.csv",                 "22_Track_B_RF",                "READY",   "54 rows (18 cities × 3 horizons)"),
    ("outputs/track_b_gbr.csv",                "23_Track_B_GBR",               "READY",   "54 rows — CANONICAL"),
    ("outputs/track_b_xgb.csv",                "24_Track_B_XGB",               "READY",   "54 rows"),
    ("outputs/track_b_lstm.csv",               "25_Track_B_LSTM",              "READY",   "54 rows — CANONICAL"),
    ("outputs/track_b_bilstm.csv",             "32_Track_B_BiLSTM",            "READY",   "54 rows — CANONICAL"),
    ("outputs/track_b_cnn_bilstm.csv",         "33_Track_B_CNN_BiLSTM",        "READY",   "54 rows"),
    ("outputs/final_track_b_complete.csv",     "28_Scientific_Comparison",     "READY",   "324 rows — master Track B table"),
    ("outputs/track_b_classical.csv",          "28_Scientific_Comparison",     "READY",   "Classical subset"),
    ("outputs/track_b_gbr_copy.csv",           "23_Track_B_GBR (Copy)(Copy)",  "EXCLUDE", "Archived copy — identical to canonical"),

    # ── Comparison & ranking CSVs
    ("outputs/final_comparison.csv",           "28_Scientific_Comparison",     "READY",   "All 450 evaluations"),
    ("outputs/track_a_model_ranking.csv",      "28_Scientific_Comparison",     "READY",   "7-model rank table"),
    ("outputs/track_b_model_ranking.csv",      "28_Scientific_Comparison",     "READY",   "6-model rank table"),
    ("outputs/track_b_horizon_ranking.csv",    "28_Scientific_Comparison",     "READY",   "t+1h/6h/24h ranking"),
    ("outputs/track_a_city_ranking.csv",       "28_Scientific_Comparison",     "READY",   "18-city ranking"),
    ("outputs/track_a_city_analysis.csv",      "27_Track_A_Audit",             "READY",   "City difficulty scores"),
    ("outputs/track_a_feature_importance.csv", "27_Track_A_Audit",             "READY",   "Top-20 features"),
    ("outputs/effect_size_analysis.csv",       "36_Effect_Size_Analysis",      "READY",   "Pairwise Δ R²"),

    # ── Leakage & audit artifacts
    ("outputs/leakage/feature_catalog.csv",    "09_Save_Leakage_Artifacts",    "READY",   "114-feature catalog"),
    ("outputs/leakage/feature_census.csv",     "11_Scientific_Validation",     "READY",   "88 safe + 26 leaky"),
    ("outputs/leakage/audit_experiments.csv",  "09_Save_Leakage_Artifacts",    "READY",   "Exp A/B results"),
    ("outputs/leakage/verdict.json",           "09_Save_Leakage_Artifacts",    "READY",   "Leakage verdict"),
    ("outputs/leakage/scientific_validation.json","11_Scientific_Validation",  "READY",   "Identity test results"),
    ("outputs/leakage/tasktype_experiments.csv","11_Scientific_Validation",    "READY",   "Estimation vs forecasting"),
    ("outputs/leakage/baseline_all_cities.csv","10_Full_Baseline_All_Cities",  "READY",   "Exp-B baseline"),
    ("outputs/leakage/best_per_city.csv",      "10_Full_Baseline_All_Cities",  "READY",   "Best model per city"),
    ("outputs/track_a_leakage_certificate.json","27_Track_A_Audit",            "READY",   "Leakage PASS certificate"),
    ("outputs/comparison_summary.json",        "28_Scientific_Comparison",     "READY",   "Summary stats"),
    ("outputs/research_verdict.json",          "29_Research_Verdict",          "READY",   "Final verdict"),
    ("outputs/block_lineage_table.csv",        "43_Block_Lineage_Audit",       "READY",   "Copy-block safety audit"),
    ("outputs/block_classification.csv",       "41_Repository_Cleanup_Report", "READY",   "All-block classification"),

    # ── Final audit CSVs
    ("outputs/final_audit/city_forensics.csv",      "P0_City_Forensics",       "READY",   "City row-count forensics"),
    ("outputs/final_audit/feature_missingness.csv", "P0_City_Forensics",       "READY",   "Per-feature missing %"),
    ("outputs/final_audit/city_feature_recovery.csv","P1_City_Recovery",       "READY",   "Recovery log"),
    ("outputs/final_audit/leakage_experiments.csv", "P2_Leakage_Experiments",  "READY",   "Exp A/B/C per city"),

    # ── Markdown reports
    ("outputs/track_a_paper_package.md",       "29_Research_Verdict",          "READY",   "Full paper package"),
    ("outputs/track_a_lstm_analysis.md",       "27_Track_A_Audit",             "READY",   "LSTM failure analysis"),
    ("outputs/feature_importance_interpretation.md","37_Feature_Importance_Interpretation","READY","Environmental narrative"),
    ("outputs/effect_size_analysis.md",        "36_Effect_Size_Analysis",      "READY",   "Effect size narrative"),
    ("outputs/deployment_recommendations.md",  "38_Deployment_Recommendations","READY",   "App A/B deployment plan"),
    ("outputs/reviewer_qa.md",                 "39_Reviewer_QA",               "READY",   "30 Q&A pairs"),
    ("outputs/final_internship_summary.md",    "40_Final_Internship_Summary",  "READY",   "1-page executive summary"),
    ("outputs/repository_cleanup_report.md",   "41_Repository_Cleanup_Report", "READY",   "Block audit report"),
    ("outputs/github_readiness_report.md",     "42_GitHub_Readiness",          "READY",   "56-file readiness check"),
    ("outputs/final_results_validation.md",    "35_Results_Validation",        "READY",   "Consistency audit"),
    ("outputs/final_block_lineage_report.md",  "43_Block_Lineage_Audit",       "READY",   "Copy-block lineage"),
    ("outputs/workspace_organization.md",      "44_Workspace_Organization",    "READY",   "Block section map"),
    ("outputs/github_structure.txt",           "42_GitHub_Readiness",          "READY",   "Repository tree"),

    # ── Publication figures (13 PNG files)
    ("outputs/comparison_figures/fig1_track_a_model_comparison.png", "28_Scientific_Comparison","READY","Track A bar chart"),
    ("outputs/comparison_figures/fig2_track_b_model_comparison.png", "28_Scientific_Comparison","READY","Track B bar chart"),
    ("outputs/comparison_figures/fig3_horizon_degradation.png",      "28_Scientific_Comparison","READY","Horizon decay curve"),
    ("outputs/comparison_figures/fig4_city_model_heatmap.png",       "28_Scientific_Comparison","READY","City × model R² heatmap"),
    ("outputs/comparison_figures/fig5_classical_vs_dl.png",          "28_Scientific_Comparison","READY","Classical vs DL"),
    ("outputs/comparison_figures/fig6_dl_comparison.png",            "28_Scientific_Comparison","READY","LSTM/BiLSTM/CNN-BiLSTM"),
    ("outputs/comparison_figures/fig7_best_vs_worst_city.png",       "28_Scientific_Comparison","READY","Best vs worst city"),
    ("outputs/comparison_figures/fig8_track_a_vs_track_b.png",       "28_Scientific_Comparison","READY","Track A vs Track B"),
    ("outputs/comparison_figures/fig9_city_difficulty.png",          "27_Track_A_Audit",         "READY","City difficulty ranking"),
    ("outputs/comparison_figures/fig10_feature_category_importance.png","37_Feature_Importance_Interpretation","READY","Feature category chart"),
    ("outputs/comparison_figures/fig11_final_certification.png",     "29_Research_Verdict",      "READY","Certification badge"),
    ("outputs/comparison_figures/fig12_feature_category_importance.png","37_Feature_Importance_Interpretation","READY","Updated feature chart"),
    ("outputs/comparison_figures/fig13_final_summary.png",           "40_Final_Internship_Summary","READY","Summary dashboard"),

    # ── Saved model
    ("outputs/lstm_diagnostic/best_delhi.keras",  "15_LSTM_Diagnostic_DelhiNCR","GIT-LFS","497 KB — Keras model file"),
    ("outputs/lstm_diagnostic/lstm_diagnostic.json","15_LSTM_Diagnostic_DelhiNCR","READY","Diagnostic metadata"),
]

# ── Stats ──────────────────────────────────────────────────────────────────
from collections import Counter
status_counts = Counter(s for _, _, s, _ in MANIFEST)

# ── Build markdown ─────────────────────────────────────────────────────────
md = []
md.append("# AQI RESEARCH PROJECT — GITHUB EXPORT MAP")
md.append("")
md.append(f"**Total files mapped:** {len(MANIFEST)}")
md.append(f"**READY:** {status_counts['READY']} | "
           f"**GIT-LFS:** {status_counts['GIT-LFS']} | "
           f"**REVIEW:** {status_counts['REVIEW']} | "
           f"**EXCLUDE:** {status_counts['EXCLUDE']}")
md.append("")
md.append("**Status legend:**")
md.append("- `READY` — commit to GitHub as-is")
md.append("- `GIT-LFS` — large binary; track with Git LFS (`git lfs track '*.parquet'`)")
md.append("- `REVIEW` — pilot/superseded; include only if completeness required")
md.append("- `EXCLUDE` — archived copy; do not commit")
md.append("")
md.append("---")
md.append("")
md.append("| Repository File | Source Block | Status | Notes |")
md.append("|----------------|-------------|--------|-------|")
for repo_file, source_block, status, notes in MANIFEST:
    status_fmt = f"**{status}**" if status in ("READY", "EXCLUDE") else f"`{status}`"
    md.append(f"| `{repo_file}` | `{source_block}` | {status_fmt} | {notes} |")

md.append("")
md.append("---")
md.append("")
md.append("## RECOMMENDED .gitattributes (Git LFS)")
md.append("")
md.append("```")
md.append("*.parquet filter=lfs diff=lfs merge=lfs -text")
md.append("*.npz     filter=lfs diff=lfs merge=lfs -text")
md.append("*.keras   filter=lfs diff=lfs merge=lfs -text")
md.append("*.pkl     filter=lfs diff=lfs merge=lfs -text")
md.append("```")
md.append("")
md.append("## FILES TO EXCLUDE FROM COMMIT (.gitignore)")
md.append("")
md.append("```")
md.append("# Raw data (too large, not redistributable)")
md.append("CPCB_Data/")
md.append("# Preprocessed numpy arrays (large; regenerate from pipeline)")
md.append("outputs/preprocessed/*.npz")
md.append("# Archived copy artifact")
md.append("outputs/track_b_gbr_copy.csv")
md.append("# Python cache")
md.append("__pycache__/")
md.append("*.pyc")
md.append("```")

md_text = "\n".join(md)
out_path = OUT / "github_export_map.md"
with open(out_path, "w") as f:
    f.write(md_text)

# ── Console ────────────────────────────────────────────────────────────────
print(SEP)
print("  BLOCK 45 — GITHUB EXPORT MAP")
print(SEP)
print(f"\n  Total files mapped : {len(MANIFEST)}")
for status, count in sorted(status_counts.items()):
    sym = {"READY": "✅", "GIT-LFS": "📦", "REVIEW": "🔍", "EXCLUDE": "❌"}.get(status, "•")
    print(f"    {sym}  {status:<10}  {count} files")
print(f"\n  ✓ Saved → {out_path}")
print(SEP)
