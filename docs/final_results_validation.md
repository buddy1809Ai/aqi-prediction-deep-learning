# Final Results Validation Report
**Generated:** Post-training audit — no model training performed  
**Status:** REVIEW REQUIRED  

## Track A — AQI Estimation
| Model | Rows | Cities | Duplicates | Missing Metrics | Null Values |
|-------|------|--------|------------|-----------------|-------------|
| Ridge | 18 | 18 | 0 | None | 0 |
| RandomForest | 18 | 18 | 0 | None | 0 |
| GradBoost | 18 | 18 | 0 | None | 0 |
| XGBoost | 18 | 18 | 0 | None | 0 |
| LSTM | 18 | 18 | 0 | None | 0 |
| BiLSTM | 18 | 18 | 0 | None | 0 |
| CNN-BiLSTM | 18 | 18 | 0 | None | 0 |

## Track B — AQI Forecasting
| Model | Rows | Cities | Horizons | Duplicates | Missing Metrics | Null Values |
|-------|------|--------|----------|------------|-----------------|-------------|
| RandomForest | 54 | 18 | [1, 6, 24] | 0 | None | 0 |
| GradBoost | 54 | 18 | [1, 6, 24] | 0 | None | 0 |
| XGBoost | 54 | 18 | [1, 6, 24] | 0 | None | 0 |
| LSTM | 54 | 18 | [1, 6, 24] | 0 | None | 0 |
| BiLSTM | 54 | 18 | [1, 6, 24] | 0 | None | 0 |
| CNN-BiLSTM | 54 | 18 | [1, 6, 24] | 0 | None | 0 |

## Issues Found
Total issues: 1
- Track A: 1 rows with R² < -0.5

## Final Verdict
**REVIEW REQUIRED**

- Track A: 7 models × 18 cities = 126 rows expected
- Track B: 6 models × 18 cities × 3 horizons = 324 rows expected
- No AQI-derived features in any model (confirmed by leakage certificate)
- No future information used
- Chronological splits preserved