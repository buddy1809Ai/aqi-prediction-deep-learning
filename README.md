# AQI Prediction Using Deep Learning
### A Dual-Track Research Study — CPCB Multi-City India Dataset

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-orange)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---


## Dataset

This repository contains only sample datasets for demonstration.

The complete CPCB dataset used in this research can be downloaded from:

**[📥 Download Complete CPCB Dataset — Google Drive](https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link)**

> **Replace `<GOOGLE_DRIVE_LINK>` with your actual Google Drive sharing link before publishing.**

### Dataset Details
| Property | Value |
|---|---|
| Source | Central Pollution Control Board (CPCB), India |
| Cities | 19 Indian cities |
| Records | ~18.7 million (15-minute intervals) |
| Cleaned hourly rows | ~934,775 |
| Period | 2017 – 2025 |
| Format | CSV (15-min raw) / Parquet (cleaned/engineered) |
| License | Government of India Open Data License |

### Sample Data
The `data/samples/` folder contains **500–1000 representative rows per city**
from the cleaned dataset, preserving original column names and structure.
These are suitable for running code in dry-run/demo mode.

### How to Use the Full Dataset
1. Download and unzip the archive from the Google Drive link above
2. Place the `CPCB_Data/` folder at the repository root
3. Re-run blocks 02–07 (schema inspection → cleaning → feature engineering)
4. The model training blocks (13 onwards) will then use the full data
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
