# AQI Research Project — Repository Cleanup Report

**Total blocks:** 43 | **Keep:** 36 | **Delete:** 7


## Block Classification Table

| Block Name | Category | Decision | Reason |
|---|---|---|---|
| python_block (check_library_versions) | Temporary Diagnostic | DELETE | Simple version-check stub; environment verified by 12_Environment_Verification |
| 02_Schema_Sample | Core Pipeline | KEEP | Phase 0 schema inspection; upstream of 03_Inventory_Report |
| 03_Inventory_Report | Core Pipeline | KEEP | Phase 0 dataset inventory with figures; upstream of 04_Data_Cleaning |
| 04_Data_Cleaning | Core Pipeline | KEEP | Phase 2 hourly aggregation + AQI computation; saves 18 cleaned parquets |
| 05_EDA | Core Pipeline | KEEP | Phase 3 EDA with 9 publication figures |
| 06_Feature_Engineering | Core Pipeline | KEEP | Phase 3 lag/rolling/cyclical/interaction features; saves 18 engineered parquets |
| 07_Preprocessing | Core Pipeline | KEEP | Phase 4 train/val/test split + MinMaxScaler; saves 18 npz + city_scalers.pkl |
| 09_Save_Leakage_Artifacts | Core Pipeline | KEEP | Phase 5 persists feature catalog, audit CSVs, verdict JSON |
| 10_Full_Baseline_All_Cities | Core Pipeline | KEEP | Phase 5 Exp-B honest baseline; produces baseline_all_cities.csv + figures |
| 11_Scientific_Validation | Research Analysis | KEEP | Phase 5 identity test + task-type experiments; confirms estimation vs forecasting |
| 12_Environment_Verification | Temporary Diagnostic | KEEP | Documents environment for reproducibility; referenced in paper package |
| 13_Track_A_Estimation | Core Pipeline | KEEP | Early Track A prototype; produces track_a_estimation_results.csv (unique artifact) |
| 15_LSTM_Diagnostic_DelhiNCR | Research Analysis | KEEP | Phase 4 LSTM gate; validates sequence generation; referenced in paper methodology |
| 16_Track_A_LSTM_All_Cities | Core Pipeline | KEEP | Phase 6 LSTM training 18 cities; saves final_track_a_lstm.csv + figure |
| 17_Track_A_Ridge_All_Cities | Core Pipeline | KEEP | Phase 6 Ridge 18 cities; saves track_a_ridge.csv |
| 18_Track_A_RF_All_Cities | Core Pipeline | KEEP | Phase 6 RF 18 cities; saves track_a_rf.csv |
| 19_Track_A_GBR_All_Cities | Core Pipeline | KEEP | Phase 6 GBR 18 cities; saves track_a_gbr.csv |
| 20_Track_A_XGB_All_Cities | Core Pipeline | KEEP | Phase 6 XGB 18 cities; saves track_a_xgb.csv |
| 22_Track_B_RF | Core Pipeline | KEEP | Phase 7 RF 18×3 forecasting; saves track_b_rf.csv |
| 23_Track_B_GBR | Core Pipeline | KEEP | Phase 7 GBR 18×3 forecasting; saves track_b_gbr.csv |
| 23_Track_B_GBR (Copy) (Copy) | Duplicate | DELETE | Exact duplicate; saves to track_b_gbr_copy.csv — not referenced in any merge |
| 24_Track_B_XGB (Copy) | Duplicate | DELETE | Checkpoint shows 54 rows already done from canonical block; no unique output |
| 25_Track_B_LSTM | Core Pipeline | KEEP | Phase 7 LSTM 18×3 forecasting; saves track_b_lstm.csv |
| 25_Track_B_LSTM (Copy) | Duplicate | DELETE | 100% checkpoint skips; saves to same track_b_lstm.csv as canonical block |
| 26_Track_B_Merge | Obsolete | DELETE | Never ran (null status); superseded by 28_Scientific_Comparison which does full merge |
| 27_Track_A_Audit | Research Analysis | KEEP | Phase 8 leakage cert + city difficulty + LSTM analysis; saves 5 audit files |
| 28_Scientific_Comparison | Research Analysis | KEEP | Phase 10 master merge + 8 pub figures; saves final_comparison.csv + rankings |
| 29_Research_Verdict | Research Analysis | KEEP | Phase 13 final verdict JSON + certification figure |
| 30_Track_A_BiLSTM | Core Pipeline | KEEP | Phase 6 BiLSTM 18 cities; saves track_a_bilstm.csv |
| 31_Track_A_CNN_BiLSTM | Core Pipeline | KEEP | Phase 6 CNN-BiLSTM 18 cities; saves track_a_cnn_bilstm.csv |
| 32_Track_B_BiLSTM (Copy) | Duplicate | DELETE | 100% checkpoint skips; real results already written by 32_Track_B_BiLSTM |
| 32_Track_B_BiLSTM | Core Pipeline | KEEP | Phase 7 BiLSTM 18×3 forecasting; saves track_b_bilstm.csv |
| 33_Track_B_CNN_BiLSTM | Core Pipeline | KEEP | Phase 7 CNN-BiLSTM 18×3 forecasting; saves track_b_cnn_bilstm.csv |
| 35_Results_Validation | Research Analysis | KEEP | Phase audit; saves final_results_validation.md — publication consistency check |
| 36_Effect_Size_Analysis | Research Analysis | KEEP | Quantifies GBR vs DL gaps; saves effect_size_analysis.csv + .md |
| 37_Feature_Importance_Interpretation | Research Analysis | KEEP | Environmental narrative for PM2.5 dominance; saves feature_importance_interpretation.md |
| 38_Deployment_Recommendations | Deployment | KEEP | API design specs for estimation + forecasting apps; saves deployment_recommendations.md |
| 39_Reviewer_QA | Research Analysis | KEEP | 30 reviewer Q&A pairs; saves reviewer_qa.md |
| 40_Final_Internship_Summary | Research Analysis | KEEP | 1-page executive summary + fig13; saves final_internship_summary.md |
| P0_City_Forensics | Core Pipeline | KEEP | Phase 0 forensics; saves city_forensics.csv + feature_missingness.csv |
| P1_City_Recovery | Core Pipeline | KEEP | Phase 1 recovery pipeline; saves 18 recovered parquets + city_feature_recovery.csv |
| P2_Leakage_Experiments | Core Pipeline | KEEP | Phase 5 Exp-A/B/C leakage quantification; saves leakage_experiments.csv |
| check_library_versions | Temporary Diagnostic | DELETE | Duplicate of 12_Environment_Verification; outputs sklearn/numpy/pandas versions only |

## Blocks Marked for Deletion

| Block Name | Reason |
|---|---|
| python_block (check_library_versions) | Simple version-check stub; environment verified by 12_Environment_Verification |
| 23_Track_B_GBR (Copy) (Copy) | Exact duplicate; saves to track_b_gbr_copy.csv — not referenced in any merge |
| 24_Track_B_XGB (Copy) | Checkpoint shows 54 rows already done from canonical block; no unique output |
| 25_Track_B_LSTM (Copy) | 100% checkpoint skips; saves to same track_b_lstm.csv as canonical block |
| 26_Track_B_Merge | Never ran (null status); superseded by 28_Scientific_Comparison which does full merge |
| 32_Track_B_BiLSTM (Copy) | 100% checkpoint skips; real results already written by 32_Track_B_BiLSTM |
| check_library_versions | Duplicate of 12_Environment_Verification; outputs sklearn/numpy/pandas versions only |

## Category Summary

| Category | Keep | Delete |
|---|---|---|
| Core Pipeline | 24 | 0 |
| Research Analysis | 10 | 0 |
| Deployment | 1 | 0 |
| Temporary Diagnostic | 1 | 2 |
| Duplicate | 0 | 4 |
| Obsolete | 0 | 1 |

## Recommended Pipeline Sequence
```

RECOMMENDED PIPELINE SEQUENCE (top-to-bottom)
==============================================================================
PHASE 0 — PROJECT SETUP & INVENTORY
  01. python_block           (Dataset Audit — raw CSV scan)
  02. 02_Schema_Sample       (Schema inspection)
  03. 03_Inventory_Report    (Dataset inventory + figures)

PHASE 1 — DATA PROCESSING
  04. 04_Data_Cleaning       (Hourly aggregation, AQI computation, 18 cleaned parquets)
  05. 05_EDA                 (9 publication EDA figures)
  06. 06_Feature_Engineering (Lag/rolling/cyclical features, 18 engineered parquets)
  07. 07_Preprocessing       (Train/val/test split, MinMaxScaler, 18 npz files)

PHASE 2 — CITY RECOVERY & LEAKAGE
  08. P0_City_Forensics      (Per-city missing-value census)
  09. P1_City_Recovery       (Imputation pipeline, 18 recovered parquets)
  10. P2_Leakage_Experiments (Exp-A/B/C RF experiments)
  11. 09_Save_Leakage_Artifacts (Persist leakage CSVs/JSON)
  12. 10_Full_Baseline_All_Cities (Exp-B honest baseline, 18 cities)
  13. 11_Scientific_Validation   (Identity test + task-type classification)
  14. 12_Environment_Verification (Package registry for reproducibility)

PHASE 3 — TRACK A MODELS (AQI Estimation)
  15. 13_Track_A_Estimation     (Early prototype, Ridge/RF/GBR/XGB/LSTM combined)
  16. 15_LSTM_Diagnostic_DelhiNCR (LSTM gate test)
  17. 17_Track_A_Ridge_All_Cities
  18. 18_Track_A_RF_All_Cities
  19. 19_Track_A_GBR_All_Cities
  20. 20_Track_A_XGB_All_Cities
  21. 16_Track_A_LSTM_All_Cities
  22. 30_Track_A_BiLSTM
  23. 31_Track_A_CNN_BiLSTM

PHASE 4 — TRACK B MODELS (AQI Forecasting)
  24. 22_Track_B_RF
  25. 23_Track_B_GBR
  26. 25_Track_B_LSTM
  27. 32_Track_B_BiLSTM
  28. 33_Track_B_CNN_BiLSTM

PHASE 5 — SCIENTIFIC COMPARISON & AUDIT
  29. 27_Track_A_Audit         (Leakage cert + city difficulty + LSTM analysis)
  30. 28_Scientific_Comparison (Master merge, 8 pub figures, final_comparison.csv)
  31. 29_Research_Verdict      (Final verdict JSON + certification)
  32. 35_Results_Validation    (Consistency check report)
  33. 36_Effect_Size_Analysis  (Model effect size quantification)

PHASE 6 — PUBLICATION PACKAGE
  34. 37_Feature_Importance_Interpretation
  35. 38_Deployment_Recommendations
  36. 39_Reviewer_QA
  37. 40_Final_Internship_Summary
  38. 41_Repository_Cleanup_Report     ← THIS BLOCK
  39. 42_GitHub_Readiness              ← NEXT BLOCK
  40. 43_Streamlit_Design_Document     ← NEXT BLOCK
  41. 44_Final_Pipeline_Summary        ← NEXT BLOCK
==============================================================================

```
