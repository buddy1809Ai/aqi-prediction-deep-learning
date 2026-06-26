"""
BLOCK 2 — GITHUB READINESS CHECK + REPOSITORY STRUCTURE
Verifies all outputs, generates repo skeleton docs.
NO MODEL TRAINING — pure documentation.
"""
import os
from pathlib import Path

OUT  = Path("outputs")
SEP  = "=" * 70

# ── All expected output files ─────────────────────────────────────────────────
expected_files = {
    # TRACK A
    "outputs/track_a_ridge.csv":             "Track A Ridge results (18 cities)",
    "outputs/track_a_rf.csv":                "Track A Random Forest results",
    "outputs/track_a_gbr.csv":               "Track A Gradient Boosting results",
    "outputs/track_a_xgb.csv":               "Track A XGBoost results",
    "outputs/final_track_a_lstm.csv":        "Track A LSTM results",
    "outputs/track_a_bilstm.csv":            "Track A BiLSTM results",
    "outputs/track_a_cnn_bilstm.csv":        "Track A CNN-BiLSTM results",
    "outputs/final_track_a_classical.csv":   "Track A classical merged",
    "outputs/final_track_a_complete.csv":    "Track A full merge (7 models)",
    # TRACK B
    "outputs/track_b_rf.csv":               "Track B RF results (18×3)",
    "outputs/track_b_gbr.csv":              "Track B GBR results",
    "outputs/track_b_xgb.csv":              "Track B XGB results",
    "outputs/track_b_lstm.csv":             "Track B LSTM results",
    "outputs/track_b_bilstm.csv":           "Track B BiLSTM results",
    "outputs/track_b_cnn_bilstm.csv":       "Track B CNN-BiLSTM results",
    "outputs/track_b_classical.csv":        "Track B classical merged",
    "outputs/final_track_b_complete.csv":   "Track B full merge (6 models)",
    # COMPARISON
    "outputs/final_comparison.csv":          "Master comparison table",
    "outputs/track_a_model_ranking.csv":     "Track A model ranking",
    "outputs/track_b_model_ranking.csv":     "Track B model ranking",
    "outputs/track_b_horizon_ranking.csv":   "Horizon ranking (t+1/6/24h)",
    "outputs/track_a_city_ranking.csv":      "Track A city ranking",
    "outputs/track_a_city_analysis.csv":     "City difficulty analysis",
    "outputs/track_a_feature_importance.csv":"Feature importance top-20",
    "outputs/effect_size_analysis.csv":      "Effect size GBR vs all models",
    # AUDIT / CERTIFICATION
    "outputs/track_a_leakage_certificate.json": "Leakage certification (11 checks)",
    "outputs/research_verdict.json":         "Final research verdict",
    "outputs/comparison_summary.json":       "Comparison summary JSON",
    "outputs/leakage/feature_catalog.csv":   "114-feature leakage catalog",
    "outputs/leakage/feature_census.csv":    "115-feature census with deploy flags",
    "outputs/leakage/audit_experiments.csv": "Exp A/B/C leakage audit results",
    "outputs/leakage/verdict.json":          "Leakage verdict JSON",
    "outputs/final_audit/city_forensics.csv":"City forensics (row counts per phase)",
    "outputs/final_audit/feature_missingness.csv": "Per-feature missingness census",
    "outputs/final_results_validation.md":   "Results consistency validation",
    "outputs/final_internship_summary.md":   "1-page executive summary",
    # PUBLICATION DOCS
    "outputs/track_a_paper_package.md":      "Full paper package (abstract→future work)",
    "outputs/track_a_lstm_analysis.md":      "LSTM failure scientific analysis",
    "outputs/effect_size_analysis.md":       "Effect size narrative",
    "outputs/feature_importance_interpretation.md": "PM2.5 dominance narrative",
    "outputs/deployment_recommendations.md": "API design for estimation + forecasting apps",
    "outputs/reviewer_qa.md":               "30 reviewer Q&A pairs",
    "outputs/repository_cleanup_report.md":  "Repository cleanup audit",
    # FIGURES
    "outputs/comparison_figures/fig1_track_a_model_comparison.png":  "Track A model R² bar chart",
    "outputs/comparison_figures/fig2_track_b_model_comparison.png":  "Track B model R² bar chart",
    "outputs/comparison_figures/fig3_horizon_degradation.png":       "Forecast horizon degradation curve",
    "outputs/comparison_figures/fig4_city_model_heatmap.png":        "City×model R² heatmap",
    "outputs/comparison_figures/fig5_classical_vs_dl.png":           "Classical vs deep learning comparison",
    "outputs/comparison_figures/fig6_dl_comparison.png":             "LSTM vs BiLSTM vs CNN-BiLSTM",
    "outputs/comparison_figures/fig7_best_vs_worst_city.png":        "Best vs worst city comparison",
    "outputs/comparison_figures/fig8_track_a_vs_track_b.png":        "Track A vs Track B comparison",
    "outputs/comparison_figures/fig9_city_difficulty.png":           "City difficulty ranking",
    "outputs/comparison_figures/fig10_feature_category_importance.png": "Feature category importance",
    "outputs/comparison_figures/fig11_final_certification.png":      "Final research certification",
    "outputs/comparison_figures/fig12_feature_category_importance.png": "Feature importance (updated)",
    "outputs/comparison_figures/fig13_final_summary.png":            "Final internship summary dashboard",
}

print(SEP)
print("  BLOCK 2 — GITHUB READINESS CHECK")
print(SEP)

found, missing = [], []
total_size_mb = 0.0
for fpath, desc in expected_files.items():
    p = Path(fpath)
    if p.exists():
        sz = p.stat().st_size / 1024
        total_size_mb += sz / 1024
        found.append((fpath, desc, f"{sz:.1f} KB"))
    else:
        missing.append((fpath, desc))

print(f"\n  Files present : {len(found)} / {len(expected_files)}")
print(f"  Files missing : {len(missing)}")
print(f"  Total size    : {total_size_mb:.1f} MB")

if missing:
    print(f"\n  ⚠ MISSING FILES:")
    for f, d in missing:
        print(f"     ✗  {f}  ({d})")
else:
    print(f"\n  ✓ All {len(expected_files)} expected files present")

# ── Directory inventory ───────────────────────────────────────────────────────
dirs = {
    "outputs/cleaned":           "18 hourly cleaned parquets",
    "outputs/engineered":        "18 feature-engineered parquets",
    "outputs/recovered":         "18 imputation-recovered parquets",
    "outputs/preprocessed":      "18 scaled npz arrays + city_scalers.pkl",
    "outputs/leakage":           "Feature catalog, audit CSVs, verdict JSON",
    "outputs/final_audit":       "City forensics + missingness census",
    "outputs/comparison_figures":"13 publication-quality PNG figures",
    "outputs/lstm_diagnostic":   "Delhi NCR diagnostic model + JSON",
}
print(f"\n  DIRECTORY INVENTORY")
print(f"  {'Directory':<40} {'Files':>5}  Description")
print(f"  {'─'*38}  ─────  {'─'*35}")
for d, desc in dirs.items():
    p = Path(d)
    n = len(list(p.glob("*"))) if p.exists() else 0
    sym = "✓" if p.exists() else "✗"
    print(f"  {sym}  {d:<38} {n:>5}  {desc}")

# ── GitHub repository structure ───────────────────────────────────────────────
repo_structure = """
AQI_RESEARCH_PROJECT/
├── README.md                          # Project overview, setup, results
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Exclude large parquets, models, npz
│
├── data/
│   └── sample_data_only/              # Small sample CSVs for reproducibility
│       └── Delhi_NCR_2023_sample.csv  # 1000-row sample (git-safe)
│
├── notebooks/                         # Jupyter-style reference notebooks
│   ├── 01_Data_Audit.md               # Phase 0 inventory documentation
│   ├── 02_Data_Cleaning.md            # Cleaning pipeline reference
│   ├── 03_Feature_Engineering.md      # Feature engineering reference
│   ├── 04_Recovery.md                 # City recovery pipeline
│   ├── 05_Leakage_Audit.md            # Leakage experiments documentation
│   ├── 06_Track_A_Models.md           # Track A model training reference
│   ├── 07_Track_B_Models.md           # Track B forecasting reference
│   ├── 08_Comparison.md               # Scientific comparison reference
│   └── 09_Paper_Results.md            # Publication results reference
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaning.py                # Hourly aggregation + AQI formula
│   │   ├── feature_engineering.py    # Lag/rolling/cyclical features
│   │   └── recovery.py               # Imputation pipeline
│   ├── features/
│   │   ├── feature_catalog.py        # Feature definitions + leakage flags
│   │   └── feature_selection.py      # Track A / Track B feature sets
│   ├── models/
│   │   ├── baseline_models.py        # Ridge, RF, GBR, XGB training
│   │   ├── lstm_model.py             # LSTM architecture + training loop
│   │   ├── bilstm_model.py           # BiLSTM architecture
│   │   └── cnn_bilstm_model.py       # CNN-BiLSTM architecture
│   ├── evaluation/
│   │   ├── metrics.py                # R², MAE, RMSE, inference time
│   │   ├── leakage_audit.py          # Leakage certification logic
│   │   └── comparison.py             # Model ranking + effect size
│   └── deployment/
│       ├── estimation_app.py         # Track A GradBoost inference
│       └── forecasting_app.py        # Track B GradBoost 3-horizon inference
│
├── outputs/
│   ├── figures/  → comparison_figures/  (13 publication PNGs)
│   └── reports/  → all .md and .csv files
│
├── streamlit_app/
│   ├── app.py                        # Main Streamlit entry point
│   ├── pages/
│   │   ├── 1_AQI_Estimation.py       # Track A page
│   │   └── 2_AQI_Forecasting.py      # Track B page
│   └── utils/
│       └── aqi_utils.py              # AQI category + health recommendations
│
└── docs/
    ├── paper_package.md              # Full paper draft
    ├── reviewer_qa.md                # 30 reviewer Q&A
    └── internship_summary.md         # 1-page executive summary
"""

print(f"\n{SEP}")
print("  GITHUB REPOSITORY STRUCTURE")
print(SEP)
print(repo_structure)

# ── .gitignore ───────────────────────────────────────────────────────────────
gitignore = """# Large data files — do not commit
outputs/cleaned/*.parquet
outputs/engineered/*.parquet
outputs/recovered/*.parquet
outputs/preprocessed/*.npz
outputs/preprocessed/*.pkl
outputs/lstm_diagnostic/*.keras

# Python
__pycache__/
*.py[cod]
.env
.venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
"""

# ── requirements.txt ─────────────────────────────────────────────────────────
requirements = """# AQI Research Project — Python Dependencies
# Tested with Python 3.11

numpy==2.4.6
pandas==2.3.3
scikit-learn==1.9.0
xgboost==2.0.3
tensorflow==2.15.0
keras==3.0.0
h5py>=3.9.0
matplotlib==3.11.0
scipy==1.17.1
pyarrow==24.0.0
streamlit>=1.28.0
plotly>=5.18.0
"""

# ── README skeleton ───────────────────────────────────────────────────────────
readme = """# AQI Prediction Using Deep Learning
### A Dual-Track Research Study — CPCB Multi-City India Dataset

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

This project presents a **leakage-audited dual-track framework** for AQI prediction
across 18 Indian cities using the CPCB multi-city dataset (~934,775 hourly records).

| Track | Task | Best Model | Avg R² |
|---|---|---|---|
| **Track A** | AQI Estimation from concurrent measurements | GradBoost | 0.9906 |
| **Track B** | True AQI Forecasting (t+1h/t+6h/t+24h) | GradBoost | 0.4997 |

## Key Findings

1. **Classical ML outperforms Deep Learning** on both tracks for tabular environmental data
2. **LSTM underperforms** (Track A R²=0.641) because same-timestamp tabular features
   carry the full AQI signal — sequential modeling provides no additional benefit
3. **R²≈0.99 in Track A is NOT target leakage** — it reflects that AQI is a
   deterministic piecewise-linear formula of concurrent PM2.5/PM10/NO2 measurements
   (confirmed by identity test: formula-recomputed R²=1.0000)
4. **True forecasting (Track B) is significantly harder**: t+1h R²=0.531,
   t+6h R²=0.178, t+24h R²=0.105 — demonstrating genuine temporal uncertainty

## Results Summary

### Track A — AQI Estimation (18 cities)
| Rank | Model      | Avg R² | Avg MAE | Avg RMSE |
|------|-----------|--------|---------|----------|
| 1    | GradBoost  | 0.9906 | 2.94    | 5.77     |
| 2    | RandomForest | 0.9874 | 1.64  | 6.05     |
| 3    | XGBoost   | 0.9856 | 2.83    | 6.82     |
| 4    | Ridge     | 0.8304 | 18.69   | 28.17    |
| 5    | LSTM      | 0.6411 | 27.04   | 39.36    |
| 6    | BiLSTM    | 0.5897 | 27.57   | 40.59    |
| 7    | CNN-BiLSTM | 0.2756 | 42.25  | 58.54    |

### Track B — AQI Forecasting (18 cities × 3 horizons)
| Rank | Model      | Avg R² | Avg MAE | Avg RMSE |
|------|-----------|--------|---------|----------|
| 1    | GradBoost  | 0.4997 | 32.57   | 48.37    |
| 2    | RandomForest | 0.4914 | 34.16 | 48.79    |
| 3    | XGBoost   | 0.4902 | 32.97   | 48.86    |
| 4    | BiLSTM    | 0.2831 | 39.07   | 56.65    |
| 5    | LSTM      | 0.2768 | 39.07   | 56.83    |
| 6    | CNN-BiLSTM | −0.4147 | 48.21 | 67.50    |

## Setup

```bash
git clone https://github.com/your-username/AQI_RESEARCH_PROJECT
cd AQI_RESEARCH_PROJECT
pip install -r requirements.txt
```

## Recommended Paper Title

*"AQI Estimation vs. True Forecasting: A Leakage-Audited Dual-Track Benchmark
Across 18 Indian Cities Using Gradient Boosting and LSTM"*

**Target venue:** Environmental Modelling & Software (Elsevier, IF≈4.5)

## Project Structure

See `docs/` for full paper package, reviewer Q&A, and internship summary.

## License

MIT License — see [LICENSE](LICENSE)
"""

# ── Save all docs ─────────────────────────────────────────────────────────────
(OUT / "github_structure.txt").write_text(repo_structure)
(OUT / ".gitignore_template").write_text(gitignore)
(OUT / "requirements.txt").write_text(requirements)
(OUT / "README_template.md").write_text(readme)

# ── Readiness report ──────────────────────────────────────────────────────────
status = "✓ GITHUB READY" if not missing else f"⚠ {len(missing)} FILES MISSING"

readiness_md = f"""# GitHub Readiness Report

**Status: {status}**
**Files checked:** {len(expected_files)}
**Present:** {len(found)}
**Missing:** {len(missing)}
**Total size:** {total_size_mb:.1f} MB

## File Inventory

| File | Description | Size |
|---|---|---|
"""
for fpath, desc, sz in found:
    readiness_md += f"| `{fpath}` | {desc} | {sz} |\n"

if missing:
    readiness_md += "\n## Missing Files\n\n"
    for fpath, desc in missing:
        readiness_md += f"- `{fpath}` — {desc}\n"

readiness_md += f"""
## .gitignore Notes

Large files that should NOT be committed to GitHub:
- `outputs/cleaned/*.parquet` (~41 MB total)
- `outputs/engineered/*.parquet` (~252 MB total)
- `outputs/recovered/*.parquet` (~213 MB total)
- `outputs/preprocessed/*.npz` (~232 MB total)
- `outputs/lstm_diagnostic/*.keras` (~0.5 MB)

Safe to commit:
- All CSV result files (~100 KB total)
- All Markdown reports (~60 KB total)
- All PNG figures (~1 MB total)
- JSON audit files (~10 KB total)

## Reproducibility Notes

1. All 18 cleaned parquets can be regenerated from raw CPCB data via `04_Data_Cleaning`
2. All 18 engineered parquets regenerate via `06_Feature_Engineering`
3. All model results regenerate from recovered parquets
4. Random seed = 42 used throughout
5. Chronological 70/15/15 split used throughout — no random shuffling
"""

(OUT / "github_readiness_report.md").write_text(readiness_md)

print(f"\n{SEP}")
print(f"  GITHUB READINESS: {status}")
print(f"  Files present: {len(found)} / {len(expected_files)}")
print(f"  Total tracked output size: {total_size_mb:.1f} MB")
print(SEP)
print(f"\n  ✓ Saved → outputs/github_readiness_report.md")
print(f"  ✓ Saved → outputs/github_structure.txt")
print(f"  ✓ Saved → outputs/requirements.txt")
print(f"  ✓ Saved → outputs/README_template.md")
