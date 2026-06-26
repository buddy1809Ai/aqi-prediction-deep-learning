
"""
BLOCK 44 — WORKSPACE ORGANIZATION
Produces outputs/workspace_organization.md mapping all 39 canvas blocks
into 6 logical sections (A–F). No model training. No deletions.
"""
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

SEP = "=" * 70

SECTIONS = {
    "A — DATA AUDIT & RECOVERY": [
        ("python_block",            "Core Pipeline",   "Phase 0 — Dataset inventory across 19 cities"),
        ("02_Schema_Sample",        "Core Pipeline",   "Phase 0 — Schema inspection & column census"),
        ("03_Inventory_Report",     "Core Pipeline",   "Phase 0 — Inventory report + year-coverage heatmap"),
        ("04_Data_Cleaning",        "Core Pipeline",   "Phase 1 — Cleaning: dedup, outliers, hourly aggregation, AQI compute"),
        ("05_EDA",                  "Core Pipeline",   "Phase 2 — 9 EDA charts: distributions, trends, correlations"),
        ("06_Feature_Engineering",  "Core Pipeline",   "Phase 3 — Lag/rolling/cyclical/interaction feature creation"),
        ("07_Preprocessing",        "Core Pipeline",   "Phase 4 — Train/val/test split, MinMaxScaler (train-only fit)"),
        ("09_Save_Leakage_Artifacts","Core Pipeline",  "Phase 4b — Persist leakage catalog & audit artifacts"),
        ("10_Full_Baseline_All_Cities","Core Pipeline","Phase 5 — Exp-B honest baseline (Ridge/RF/GBR) all 18 cities"),
        ("11_Scientific_Validation","Core Pipeline",   "Phase 5b — Identity test + task-type validation (estimation vs forecasting)"),
        ("P0_City_Forensics",       "Core Pipeline",   "Phase 6 — Row-count forensics: cleaned vs engineered vs recovered"),
        ("P1_City_Recovery",        "Core Pipeline",   "Phase 6b — Drop >95%-miss features, impute, save recovered parquets"),
        ("P2_Leakage_Experiments",  "Core Pipeline",   "Phase 6c — Exp-A/B/C leakage experiment on recovered data"),
    ],
    "B — TRACK A: AQI ESTIMATION": [
        ("12_Environment_Verification","Research Analysis","Verify TF/sklearn/xgboost before training"),
        ("13_Track_A_Estimation",    "Core Pipeline",   "Track A pilot — 5 models, 3 cities (superseded by split blocks)"),
        ("15_LSTM_Diagnostic_DelhiNCR","Core Pipeline", "LSTM diagnostic — Delhi NCR gate check (R² > 0.70)"),
        ("17_Track_A_Ridge_All_Cities","Core Pipeline", "Track A — Ridge, all 18 cities → track_a_ridge.csv"),
        ("18_Track_A_RF_All_Cities", "Core Pipeline",   "Track A — Random Forest, all 18 cities → track_a_rf.csv"),
        ("19_Track_A_GBR_All_Cities","Core Pipeline",   "Track A — Gradient Boosting, all 18 cities → track_a_gbr.csv"),
        ("20_Track_A_XGB_All_Cities","Core Pipeline",   "Track A — XGBoost, all 18 cities → track_a_xgb.csv"),
        ("16_Track_A_LSTM_All_Cities","Core Pipeline",  "Track A — LSTM (seq=24), all 18 cities → final_track_a_lstm.csv"),
        ("30_Track_A_BiLSTM",        "Core Pipeline",   "Track A — BiLSTM, all 18 cities → track_a_bilstm.csv"),
        ("31_Track_A_CNN_BiLSTM",    "Core Pipeline",   "Track A — CNN-BiLSTM, all 18 cities → track_a_cnn_bilstm.csv"),
    ],
    "C — TRACK B: AQI FORECASTING": [
        ("22_Track_B_RF",            "Core Pipeline",   "Track B — RF, 18 cities × 3 horizons → track_b_rf.csv"),
        ("23_Track_B_GBR",           "Core Pipeline",   "Track B — GBR, 18 cities × 3 horizons → track_b_gbr.csv  ★ CANONICAL"),
        ("24_Track_B_XGB",           "Core Pipeline",   "Track B — XGBoost, 18 cities × 3 horizons → track_b_xgb.csv"),  # note: block name from canvas
        ("25_Track_B_LSTM",          "Core Pipeline",   "Track B — LSTM, 18 cities × 3 horizons → track_b_lstm.csv  ★ CANONICAL"),
        ("32_Track_B_BiLSTM",        "Core Pipeline",   "Track B — BiLSTM, 18 cities × 3 horizons → track_b_bilstm.csv  ★ CANONICAL"),
        ("33_Track_B_CNN_BiLSTM",    "Core Pipeline",   "Track B — CNN-BiLSTM, 18 cities × 3 horizons → track_b_cnn_bilstm.csv"),
    ],
    "D — SCIENTIFIC ANALYSIS": [
        ("27_Track_A_Audit",         "Research Analysis","Track A leakage cert, city difficulty, LSTM failure analysis"),
        ("28_Scientific_Comparison", "Research Analysis","Master comparison: 7 models × 2 tracks, 8 publication figures"),
        ("29_Research_Verdict",      "Research Analysis","Final verdict JSON, 10-question scientific audit"),
        ("35_Results_Validation",    "Research Analysis","Row-count, duplicate, metric-range consistency check"),
        ("36_Effect_Size_Analysis",  "Research Analysis","Pairwise Δ R², percentage improvement, DL vs classical gap"),
        ("37_Feature_Importance_Interpretation","Research Analysis","PM2.5 dominance narrative, environmental interpretation"),
        ("39_Reviewer_QA",           "Research Analysis","30 reviewer Q&A pairs (6 topic categories)"),
        ("43_Block_Lineage_Audit",   "Research Analysis","Copy-block safety audit — lineage table, master-merge verification"),
    ],
    "E — DEPLOYMENT & REPOSITORY": [
        ("38_Deployment_Recommendations","Deployment",  "App A (estimation) + App B (forecasting) API design & latency budget"),
        ("40_Final_Internship_Summary",  "Deployment",  "1-page executive summary + manifest dashboard figure"),
        ("41_Repository_Cleanup_Report", "Deployment",  "Block classification table: KEEP/DELETE with reasons"),
        ("42_GitHub_Readiness",          "Deployment",  "56-file GitHub readiness check + requirements.txt + README template"),
    ],
    "F — ARCHIVED BLOCKS (not used in paper results)": [
        ("check_library_versions",              "Archived","Print-only stub — superseded by 12_Environment_Verification"),
        ("python_block (check_library_versions)","Archived","Duplicate env check — same output as 12_Environment_Verification"),
        ("23_Track_B_GBR (Copy)(Copy)",         "Archived","Re-ran GBR; metrics IDENTICAL to canonical 23_Track_B_GBR (Δ=0.0)"),
        ("24_Track_B_XGB (Copy)",               "Archived","Checkpoint-skip only (0 new rows); canonical block is 24_Track_B_XGB"),
        ("25_Track_B_LSTM (Copy)",              "Archived","Checkpoint-skip only (0 new rows); canonical block is 25_Track_B_LSTM"),
        ("26_Track_B_Merge",                    "Archived","Never executed; merge handled by 28_Scientific_Comparison"),
        ("32_Track_B_BiLSTM (Copy)",            "Archived","Checkpoint-skip only (0 new rows); canonical block is 32_Track_B_BiLSTM"),
    ],
}

# ── Build markdown ─────────────────────────────────────────────────────────
lines = []
lines.append("# AQI PREDICTION RESEARCH PROJECT — WORKSPACE ORGANIZATION")
lines.append("")
lines.append("**Canvas:** AQI_Prediction  |  **Total blocks:** 39  |  **Sections:** 6")
lines.append("")
lines.append("> Blocks are **not deleted** — archived blocks are preserved for research traceability.")
lines.append("")

total_keep = 0
total_arch = 0

for section, blocks in SECTIONS.items():
    lines.append(f"## SECTION {section}")
    lines.append("")
    lines.append("| # | Block Name | Category | Description |")
    lines.append("|---|-----------|----------|-------------|")
    for idx, (name, cat, desc) in enumerate(blocks, 1):
        lines.append(f"| {idx} | `{name}` | {cat} | {desc} |")
        if "Archived" in cat:
            total_arch += 1
        else:
            total_keep += 1
    lines.append("")

lines.append("---")
lines.append("")
lines.append("## PIPELINE EXECUTION ORDER (canonical path)")
lines.append("")
pipeline = [
    ("Phase 0", "Data Inventory",      ["python_block", "02_Schema_Sample", "03_Inventory_Report"]),
    ("Phase 1", "Data Cleaning",       ["04_Data_Cleaning"]),
    ("Phase 2", "EDA",                 ["05_EDA"]),
    ("Phase 3", "Feature Engineering", ["06_Feature_Engineering"]),
    ("Phase 4", "Preprocessing",       ["07_Preprocessing", "09_Save_Leakage_Artifacts"]),
    ("Phase 5", "Baseline + Validate", ["10_Full_Baseline_All_Cities", "11_Scientific_Validation"]),
    ("Phase 6", "Recovery + Leakage",  ["P0_City_Forensics", "P1_City_Recovery", "P2_Leakage_Experiments"]),
    ("Phase 7", "Env + Diagnostic",    ["12_Environment_Verification", "15_LSTM_Diagnostic_DelhiNCR"]),
    ("Phase 8", "Track A Classical",   ["17_Track_A_Ridge_All_Cities", "18_Track_A_RF_All_Cities",
                                        "19_Track_A_GBR_All_Cities", "20_Track_A_XGB_All_Cities"]),
    ("Phase 9", "Track A Deep Learning",["16_Track_A_LSTM_All_Cities", "30_Track_A_BiLSTM", "31_Track_A_CNN_BiLSTM"]),
    ("Phase 10","Track B Classical",   ["22_Track_B_RF", "23_Track_B_GBR", "24_Track_B_XGB"]),
    ("Phase 11","Track B Deep Learning",["25_Track_B_LSTM", "32_Track_B_BiLSTM", "33_Track_B_CNN_BiLSTM"]),
    ("Phase 12","Analysis",            ["27_Track_A_Audit", "28_Scientific_Comparison", "29_Research_Verdict"]),
    ("Phase 13","Audit & Publication", ["35_Results_Validation", "36_Effect_Size_Analysis",
                                        "37_Feature_Importance_Interpretation", "39_Reviewer_QA", "43_Block_Lineage_Audit"]),
    ("Phase 14","Deployment & Repo",   ["38_Deployment_Recommendations", "40_Final_Internship_Summary",
                                        "41_Repository_Cleanup_Report", "42_GitHub_Readiness"]),
    ("Phase 15","Final Org",           ["44_Workspace_Organization", "45_GitHub_Export_Map",
                                        "46_Streamlit_Design", "47_GitHub_Push_Guide", "48_Final_Readiness_Certificate"]),
]

for phase, label, blocks in pipeline:
    block_str = " → ".join(f"`{b}`" for b in blocks)
    lines.append(f"**{phase} — {label}:** {block_str}")
    lines.append("")

lines.append("---")
lines.append("")
lines.append("## ARCHIVED BLOCKS — LINEAGE VERIFIED")
lines.append("")
lines.append("| Block | Reason Archived | Safe to Delete |")
lines.append("|-------|----------------|----------------|")
for name, cat, desc in SECTIONS["F — ARCHIVED BLOCKS (not used in paper results)"]:
    lines.append(f"| `{name}` | {desc} | ✅ YES (but retained for traceability) |")
lines.append("")
lines.append("> **Decision:** All archived blocks are retained per research traceability mandate.")
lines.append("> No block was deleted from this workspace.")
lines.append("")
lines.append(f"**Summary:** {total_keep} active blocks | {total_arch} archived blocks | 0 deleted")
lines.append("")

md_text = "\n".join(lines)
out_path = OUT / "workspace_organization.md"
with open(out_path, "w") as f:
    f.write(md_text)

# ── Console summary ────────────────────────────────────────────────────────
print(SEP)
print("  BLOCK 44 — WORKSPACE ORGANIZATION")
print(SEP)
for section, blocks in SECTIONS.items():
    sym = "📁" if "ARCHIVED" in section else "✅"
    print(f"\n  {sym}  SECTION {section}  ({len(blocks)} blocks)")
    for name, cat, desc in blocks:
        tag = "🗄️  ARCHIVED" if cat == "Archived" else f"   [{cat}]"
        print(f"       {tag}  {name}")

print(f"\n  Active blocks  : {total_keep}")
print(f"  Archived blocks: {total_arch}")
print(f"  Deleted blocks : 0")
print(f"\n  ✓ Saved → {out_path}")
print(SEP)
