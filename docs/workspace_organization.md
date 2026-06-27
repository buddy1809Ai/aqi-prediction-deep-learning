# AQI PREDICTION RESEARCH PROJECT — WORKSPACE ORGANIZATION

**Canvas:** AQI_Prediction  |  **Total blocks:** 39  |  **Sections:** 6

> Blocks are **not deleted** — archived blocks are preserved for research traceability.

## SECTION A — DATA AUDIT & RECOVERY

| # | Block Name | Category | Description |
|---|-----------|----------|-------------|
| 1 | `python_block` | Core Pipeline | Phase 0 — Dataset inventory across 19 cities |
| 2 | `02_Schema_Sample` | Core Pipeline | Phase 0 — Schema inspection & column census |
| 3 | `03_Inventory_Report` | Core Pipeline | Phase 0 — Inventory report + year-coverage heatmap |
| 4 | `04_Data_Cleaning` | Core Pipeline | Phase 1 — Cleaning: dedup, outliers, hourly aggregation, AQI compute |
| 5 | `05_EDA` | Core Pipeline | Phase 2 — 9 EDA charts: distributions, trends, correlations |
| 6 | `06_Feature_Engineering` | Core Pipeline | Phase 3 — Lag/rolling/cyclical/interaction feature creation |
| 7 | `07_Preprocessing` | Core Pipeline | Phase 4 — Train/val/test split, MinMaxScaler (train-only fit) |
| 8 | `09_Save_Leakage_Artifacts` | Core Pipeline | Phase 4b — Persist leakage catalog & audit artifacts |
| 9 | `10_Full_Baseline_All_Cities` | Core Pipeline | Phase 5 — Exp-B honest baseline (Ridge/RF/GBR) all 18 cities |
| 10 | `11_Scientific_Validation` | Core Pipeline | Phase 5b — Identity test + task-type validation (estimation vs forecasting) |
| 11 | `P0_City_Forensics` | Core Pipeline | Phase 6 — Row-count forensics: cleaned vs engineered vs recovered |
| 12 | `P1_City_Recovery` | Core Pipeline | Phase 6b — Drop >95%-miss features, impute, save recovered parquets |
| 13 | `P2_Leakage_Experiments` | Core Pipeline | Phase 6c — Exp-A/B/C leakage experiment on recovered data |

## SECTION B — TRACK A: AQI ESTIMATION

| # | Block Name | Category | Description |
|---|-----------|----------|-------------|
| 1 | `12_Environment_Verification` | Research Analysis | Verify TF/sklearn/xgboost before training |
| 2 | `13_Track_A_Estimation` | Core Pipeline | Track A pilot — 5 models, 3 cities (superseded by split blocks) |
| 3 | `15_LSTM_Diagnostic_DelhiNCR` | Core Pipeline | LSTM diagnostic — Delhi NCR gate check (R² > 0.70) |
| 4 | `17_Track_A_Ridge_All_Cities` | Core Pipeline | Track A — Ridge, all 18 cities → track_a_ridge.csv |
| 5 | `18_Track_A_RF_All_Cities` | Core Pipeline | Track A — Random Forest, all 18 cities → track_a_rf.csv |
| 6 | `19_Track_A_GBR_All_Cities` | Core Pipeline | Track A — Gradient Boosting, all 18 cities → track_a_gbr.csv |
| 7 | `20_Track_A_XGB_All_Cities` | Core Pipeline | Track A — XGBoost, all 18 cities → track_a_xgb.csv |
| 8 | `16_Track_A_LSTM_All_Cities` | Core Pipeline | Track A — LSTM (seq=24), all 18 cities → final_track_a_lstm.csv |
| 9 | `30_Track_A_BiLSTM` | Core Pipeline | Track A — BiLSTM, all 18 cities → track_a_bilstm.csv |
| 10 | `31_Track_A_CNN_BiLSTM` | Core Pipeline | Track A — CNN-BiLSTM, all 18 cities → track_a_cnn_bilstm.csv |

## SECTION C — TRACK B: AQI FORECASTING

| # | Block Name | Category | Description |
|---|-----------|----------|-------------|
| 1 | `22_Track_B_RF` | Core Pipeline | Track B — RF, 18 cities × 3 horizons → track_b_rf.csv |
| 2 | `23_Track_B_GBR` | Core Pipeline | Track B — GBR, 18 cities × 3 horizons → track_b_gbr.csv  ★ CANONICAL |
| 3 | `24_Track_B_XGB` | Core Pipeline | Track B — XGBoost, 18 cities × 3 horizons → track_b_xgb.csv |
| 4 | `25_Track_B_LSTM` | Core Pipeline | Track B — LSTM, 18 cities × 3 horizons → track_b_lstm.csv  ★ CANONICAL |
| 5 | `32_Track_B_BiLSTM` | Core Pipeline | Track B — BiLSTM, 18 cities × 3 horizons → track_b_bilstm.csv  ★ CANONICAL |
| 6 | `33_Track_B_CNN_BiLSTM` | Core Pipeline | Track B — CNN-BiLSTM, 18 cities × 3 horizons → track_b_cnn_bilstm.csv |

## SECTION D — SCIENTIFIC ANALYSIS

| # | Block Name | Category | Description |
|---|-----------|----------|-------------|
| 1 | `27_Track_A_Audit` | Research Analysis | Track A leakage cert, city difficulty, LSTM failure analysis |
| 2 | `28_Scientific_Comparison` | Research Analysis | Master comparison: 7 models × 2 tracks, 8 publication figures |
| 3 | `29_Research_Verdict` | Research Analysis | Final verdict JSON, 10-question scientific audit |
| 4 | `35_Results_Validation` | Research Analysis | Row-count, duplicate, metric-range consistency check |
| 5 | `36_Effect_Size_Analysis` | Research Analysis | Pairwise Δ R², percentage improvement, DL vs classical gap |
| 6 | `37_Feature_Importance_Interpretation` | Research Analysis | PM2.5 dominance narrative, environmental interpretation |
| 7 | `39_Reviewer_QA` | Research Analysis | 30 reviewer Q&A pairs (6 topic categories) |
| 8 | `43_Block_Lineage_Audit` | Research Analysis | Copy-block safety audit — lineage table, master-merge verification |

## SECTION E — DEPLOYMENT & REPOSITORY

| # | Block Name | Category | Description |
|---|-----------|----------|-------------|
| 1 | `38_Deployment_Recommendations` | Deployment | App A (estimation) + App B (forecasting) API design & latency budget |
| 2 | `40_Final_Internship_Summary` | Deployment | 1-page executive summary + manifest dashboard figure |
| 3 | `41_Repository_Cleanup_Report` | Deployment | Block classification table: KEEP/DELETE with reasons |
| 4 | `42_GitHub_Readiness` | Deployment | 56-file GitHub readiness check + requirements.txt + README template |

## SECTION F — ARCHIVED BLOCKS (not used in paper results)

| # | Block Name | Category | Description |
|---|-----------|----------|-------------|
| 1 | `check_library_versions` | Archived | Print-only stub — superseded by 12_Environment_Verification |
| 2 | `python_block (check_library_versions)` | Archived | Duplicate env check — same output as 12_Environment_Verification |
| 3 | `23_Track_B_GBR (Copy)(Copy)` | Archived | Re-ran GBR; metrics IDENTICAL to canonical 23_Track_B_GBR (Δ=0.0) |
| 4 | `24_Track_B_XGB (Copy)` | Archived | Checkpoint-skip only (0 new rows); canonical block is 24_Track_B_XGB |
| 5 | `25_Track_B_LSTM (Copy)` | Archived | Checkpoint-skip only (0 new rows); canonical block is 25_Track_B_LSTM |
| 6 | `26_Track_B_Merge` | Archived | Never executed; merge handled by 28_Scientific_Comparison |
| 7 | `32_Track_B_BiLSTM (Copy)` | Archived | Checkpoint-skip only (0 new rows); canonical block is 32_Track_B_BiLSTM |

---

## PIPELINE EXECUTION ORDER (canonical path)

**Phase 0 — Data Inventory:** `python_block` → `02_Schema_Sample` → `03_Inventory_Report`

**Phase 1 — Data Cleaning:** `04_Data_Cleaning`

**Phase 2 — EDA:** `05_EDA`

**Phase 3 — Feature Engineering:** `06_Feature_Engineering`

**Phase 4 — Preprocessing:** `07_Preprocessing` → `09_Save_Leakage_Artifacts`

**Phase 5 — Baseline + Validate:** `10_Full_Baseline_All_Cities` → `11_Scientific_Validation`

**Phase 6 — Recovery + Leakage:** `P0_City_Forensics` → `P1_City_Recovery` → `P2_Leakage_Experiments`

**Phase 7 — Env + Diagnostic:** `12_Environment_Verification` → `15_LSTM_Diagnostic_DelhiNCR`

**Phase 8 — Track A Classical:** `17_Track_A_Ridge_All_Cities` → `18_Track_A_RF_All_Cities` → `19_Track_A_GBR_All_Cities` → `20_Track_A_XGB_All_Cities`

**Phase 9 — Track A Deep Learning:** `16_Track_A_LSTM_All_Cities` → `30_Track_A_BiLSTM` → `31_Track_A_CNN_BiLSTM`

**Phase 10 — Track B Classical:** `22_Track_B_RF` → `23_Track_B_GBR` → `24_Track_B_XGB`

**Phase 11 — Track B Deep Learning:** `25_Track_B_LSTM` → `32_Track_B_BiLSTM` → `33_Track_B_CNN_BiLSTM`

**Phase 12 — Analysis:** `27_Track_A_Audit` → `28_Scientific_Comparison` → `29_Research_Verdict`

**Phase 13 — Audit & Publication:** `35_Results_Validation` → `36_Effect_Size_Analysis` → `37_Feature_Importance_Interpretation` → `39_Reviewer_QA` → `43_Block_Lineage_Audit`

**Phase 14 — Deployment & Repo:** `38_Deployment_Recommendations` → `40_Final_Internship_Summary` → `41_Repository_Cleanup_Report` → `42_GitHub_Readiness`

**Phase 15 — Final Org:** `44_Workspace_Organization` → `45_GitHub_Export_Map` → `46_Streamlit_Design` → `47_GitHub_Push_Guide` → `48_Final_Readiness_Certificate`

---

## ARCHIVED BLOCKS — LINEAGE VERIFIED

| Block | Reason Archived | Safe to Delete |
|-------|----------------|----------------|
| `check_library_versions` | Print-only stub — superseded by 12_Environment_Verification | ✅ YES (but retained for traceability) |
| `python_block (check_library_versions)` | Duplicate env check — same output as 12_Environment_Verification | ✅ YES (but retained for traceability) |
| `23_Track_B_GBR (Copy)(Copy)` | Re-ran GBR; metrics IDENTICAL to canonical 23_Track_B_GBR (Δ=0.0) | ✅ YES (but retained for traceability) |
| `24_Track_B_XGB (Copy)` | Checkpoint-skip only (0 new rows); canonical block is 24_Track_B_XGB | ✅ YES (but retained for traceability) |
| `25_Track_B_LSTM (Copy)` | Checkpoint-skip only (0 new rows); canonical block is 25_Track_B_LSTM | ✅ YES (but retained for traceability) |
| `26_Track_B_Merge` | Never executed; merge handled by 28_Scientific_Comparison | ✅ YES (but retained for traceability) |
| `32_Track_B_BiLSTM (Copy)` | Checkpoint-skip only (0 new rows); canonical block is 32_Track_B_BiLSTM | ✅ YES (but retained for traceability) |

> **Decision:** All archived blocks are retained per research traceability mandate.
> No block was deleted from this workspace.

**Summary:** 41 active blocks | 7 archived blocks | 0 deleted
