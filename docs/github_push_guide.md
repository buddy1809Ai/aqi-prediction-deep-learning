# GITHUB PUSH GUIDE
## AQI Prediction Using Deep Learning — Repository Publication Guide

---

## 1. RECOMMENDED REPOSITORY METADATA

| Field | Value |
|-------|-------|
| **Repository name** | `aqi-prediction-deep-learning` |
| **Description** | Multi-city AQI Estimation & Forecasting using GradBoost and LSTM on CPCB India data. Dual-track leakage-audited benchmark across 18 cities. |
| **Visibility** | Public (research) |
| **License** | MIT |
| **Topics/Tags** | `air-quality` `aqi` `deep-learning` `lstm` `gradient-boosting` `india` `cpcb` `time-series-forecasting` `environmental-ml` `xgboost` |

---

## 2. FINAL REPOSITORY TREE

```
aqi-prediction-deep-learning/
│
├── README.md                        # Project overview + results summary
├── requirements.txt                 # 10 pinned packages
├── .gitignore                       # Excludes raw data, npz, __pycache__
├── .gitattributes                   # Git LFS for *.parquet, *.keras, *.pkl
│
├── notebooks/                       # Zerve canvas exported notebooks
│   └── AQI_Prediction_Canvas.pdf    # (export from Zerve if available)
│
├── outputs/
│   ├── track_a_ridge.csv
│   ├── track_a_rf.csv
│   ├── track_a_gbr.csv
│   ├── track_a_xgb.csv
│   ├── final_track_a_lstm.csv
│   ├── track_a_bilstm.csv
│   ├── track_a_cnn_bilstm.csv
│   ├── final_track_a_complete.csv   # ← 126 rows master table
│   ├── track_b_rf.csv
│   ├── track_b_gbr.csv
│   ├── track_b_xgb.csv
│   ├── track_b_lstm.csv
│   ├── track_b_bilstm.csv
│   ├── track_b_cnn_bilstm.csv
│   ├── final_track_b_complete.csv   # ← 324 rows master table
│   ├── final_comparison.csv         # ← 450 evaluations
│   ├── track_a_model_ranking.csv
│   ├── track_b_model_ranking.csv
│   ├── track_b_horizon_ranking.csv
│   ├── track_a_leakage_certificate.json
│   ├── research_verdict.json
│   ├── comparison_summary.json
│   ├── track_a_paper_package.md     # ← Full paper draft
│   ├── reviewer_qa.md               # ← 30 Q&A pairs
│   ├── deployment_recommendations.md
│   ├── final_internship_summary.md
│   ├── workspace_organization.md
│   ├── github_export_map.md
│   ├── streamlit_design_document.md
│   ├── github_readiness_report.md
│   ├── github_structure.txt
│   ├── requirements.txt
│   ├── leakage/                     # Leakage audit artifacts (6 files)
│   ├── final_audit/                 # City forensics (4 files)
│   └── comparison_figures/          # 13 publication PNG figures
│       ├── fig1_track_a_model_comparison.png
│       ├── fig2_track_b_model_comparison.png
│       ├── fig3_horizon_degradation.png
│       ├── fig4_city_model_heatmap.png
│       ├── fig5_classical_vs_dl.png
│       ├── fig6_dl_comparison.png
│       ├── fig7_best_vs_worst_city.png
│       ├── fig8_track_a_vs_track_b.png
│       ├── fig9_city_difficulty.png
│       ├── fig10_feature_category_importance.png
│       ├── fig11_final_certification.png
│       ├── fig12_feature_category_importance.png
│       └── fig13_final_summary.png
│
├── docs/
│   └── github_push_guide.md         # ← This file
│
└── streamlit_app/                   # (to be built — see 46_Streamlit_Design)
    └── README.md
```

---

## 3. .gitignore CONTENTS

```gitignore
# Raw CPCB data — not redistributable, too large
CPCB_Data/
data/raw/

# Large preprocessed arrays — regenerate from pipeline
outputs/preprocessed/
outputs/engineered/

# Archived copy artifact — not canonical
outputs/track_b_gbr_copy.csv

# Python cache
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Secrets / environment
.env
*.key

# OS files
.DS_Store
Thumbs.db
```

---

## 4. .gitattributes (Git LFS)

```gitattributes
*.parquet filter=lfs diff=lfs merge=lfs -text
*.npz     filter=lfs diff=lfs merge=lfs -text
*.keras   filter=lfs diff=lfs merge=lfs -text
*.pkl     filter=lfs diff=lfs merge=lfs -text
*.h5      filter=lfs diff=lfs merge=lfs -text
```

---

## 5. GIT COMMANDS — STEP BY STEP

### Step 1: Initialize repository
```bash
cd /path/to/project
git init
git lfs install
```

### Step 2: Create .gitattributes for LFS
```bash
git lfs track '*.parquet'
git lfs track '*.npz'
git lfs track '*.keras'
git lfs track '*.pkl'
# This auto-creates/updates .gitattributes
```

### Step 3: Create GitHub repository
```bash
# Option A — GitHub CLI
gh repo create aqi-prediction-deep-learning \
  --public \
  --description 'Multi-city AQI Estimation & Forecasting — CPCB India dataset'

# Option B — GitHub web UI
# 1. Go to github.com → New repository
# 2. Name: aqi-prediction-deep-learning
# 3. Visibility: Public
# 4. Do NOT initialize with README (we already have one)
```

### Step 4: Add remote and stage files
```bash
git remote add origin https://github.com/YOUR_USERNAME/aqi-prediction-deep-learning.git

# Stage root files
git add README.md requirements.txt .gitignore .gitattributes

# Stage outputs (CSVs, JSONs, markdowns, PNGs)
git add outputs/*.csv outputs/*.json outputs/*.md outputs/*.txt
git add outputs/leakage/ outputs/final_audit/ outputs/comparison_figures/
git add outputs/lstm_diagnostic/lstm_diagnostic.json

# Stage docs
git add docs/

# Stage parquets via LFS (cleaned + recovered)
git add outputs/cleaned/
git add outputs/recovered/

# Verify LFS is tracking large files
git lfs status
```

### Step 5: Commit and push
```bash
git commit -m 'feat: complete AQI dual-track research project'

# Add structured commit body
git commit --amend -m "feat: complete AQI dual-track research project

- 18 Indian cities (CPCB dataset, ~18.7M records)
- Track A: AQI Estimation — 7 models, avg R²=0.99 (GradBoost)
- Track B: AQI Forecasting — 6 models × 3 horizons (t+1h/6h/24h)
- Full leakage audit — 11/11 PASS
- 450 model evaluations, 13 publication figures
- Dual-track framework ready for journal submission"

git push -u origin main
```

### Step 6: Add GitHub topics (web UI or CLI)
```bash
# GitHub CLI
gh repo edit aqi-prediction-deep-learning \
  --add-topic air-quality \
  --add-topic aqi \
  --add-topic deep-learning \
  --add-topic lstm \
  --add-topic gradient-boosting \
  --add-topic india \
  --add-topic cpcb \
  --add-topic time-series-forecasting \
  --add-topic environmental-ml \
  --add-topic xgboost
```

### Step 7: Verify push
```bash
git log --oneline -5
git lfs ls-files
gh repo view --web   # opens repo in browser
```

---

## 6. PRE-PUSH CHECKLIST

| # | Check | Status |
|---|-------|--------|
| 1 | `git lfs install` run | ☐ |
| 2 | `.gitattributes` created | ☐ |
| 3 | `CPCB_Data/` in .gitignore | ☐ |
| 4 | `outputs/preprocessed/` in .gitignore | ☐ |
| 5 | `README.md` expanded with results | ☐ |
| 6 | All 13 figures present in comparison_figures/ | ☐ |
| 7 | `final_track_a_complete.csv` has 126 rows | ☐ |
| 8 | `final_track_b_complete.csv` has 324 rows | ☐ |
| 9 | `track_a_leakage_certificate.json` PASS | ☐ |
| 10 | `requirements.txt` matches environment | ☐ |

---

## 7. RECOMMENDED PAPER CITATION PLACEHOLDER

```bibtex
@article{aqi_dual_track_2024,
  title   = {AQI Estimation vs. True Forecasting: A Leakage-Audited
             Dual-Track Benchmark Across 18 Indian Cities Using
             Gradient Boosting and LSTM},
  author  = {[Your Name]},
  journal = {Environmental Modelling \& Software},
  year    = {2024},
  note    = {Under review},
  url     = {https://github.com/YOUR_USERNAME/aqi-prediction-deep-learning}
}
```