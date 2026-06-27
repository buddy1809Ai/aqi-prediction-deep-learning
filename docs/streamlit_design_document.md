# STREAMLIT APP DESIGN DOCUMENT
## AQI Prediction Using Deep Learning — Interactive Research Dashboard

**Version:** 1.0  |  **Tracks:** A (Estimation) + B (Forecasting)  |  **Models:** 13 total

---

## 1. APPLICATION STRUCTURE

```
app/
├── main.py                   # Entry point — st.set_page_config, sidebar nav
├── pages/
│   ├── 01_Track_A_Estimation.py
│   ├── 02_Track_B_Forecasting.py
│   ├── 03_Model_Comparison.py
│   └── 04_About_Research.py
├── utils/
│   ├── model_loader.py       # Load GBR/RF/XGB/Ridge from joblib
│   ├── lstm_loader.py        # Load .keras models via TF
│   ├── aqi_utils.py          # AQI category + health advisory logic
│   ├── feature_builder.py    # Build Track A / Track B feature vectors
│   └── charts.py             # Plotly chart helpers
├── assets/
│   └── aqi_category_colors.json
└── data/
    ├── outputs/final_track_a_complete.csv   # pre-loaded results
    └── outputs/final_track_b_complete.csv
```

---

## 2. GLOBAL SIDEBAR

| Widget | Type | Values |
|--------|------|--------|
| Navigation | `st.sidebar.radio` | Track A / Track B / Comparison / About |
| City selector | `st.sidebar.selectbox` | 18 cities |
| Track selector | `st.sidebar.radio` | Estimation / Forecasting |
| Theme toggle | checkbox | Light / Dark |

---

## 3. PAGE 1 — TRACK A: AQI ESTIMATION

### Purpose
User provides same-timestamp sensor readings → all 7 models estimate AQI(t) simultaneously.

### Input Panel (left column — 35% width)

| Input Field | Widget | Range / Default |
|-------------|--------|-----------------|
| City | selectbox | 18 cities |
| PM2.5 (µg/m³) | number_input | 0–500, default 60 |
| PM10 (µg/m³) | number_input | 0–600, default 90 |
| NO2 (µg/m³) | number_input | 0–400, default 40 |
| SO2 (µg/m³) | number_input | 0–800, default 15 |
| CO (mg/m³) | number_input | 0–50, default 1.2 |
| Ozone (µg/m³) | number_input | 0–200, default 30 |
| NH3 (µg/m³) | number_input | 0–400, default 10 |
| Temperature (°C) | slider | -10–50, default 28 |
| Relative Humidity (%) | slider | 0–100, default 60 |
| Wind Speed (m/s) | slider | 0–20, default 3 |
| Hour of day | selectbox | 0–23 |
| Month | selectbox | 1–12 |

### Output Panel (right column — 65% width)

**Primary Output Table:**

```
| Model          | Predicted AQI | AQI Category  | Health Advisory               |
|----------------|--------------|---------------|-------------------------------|
| Ridge          |   142        | Moderate      | Sensitive groups reduce outdoor |
| Random Forest  |   138        | Moderate      | ...
| Gradient Boost |   140        | Moderate      | ...  ← 🏆 Best model          |
| XGBoost        |   141        | Moderate      | ...
| LSTM           |   135        | Moderate      | ...
| BiLSTM         |   133        | Moderate      | ...
| CNN-BiLSTM     |   129        | Moderate      | ...
```

**AQI Category Color Coding:**
- 🟢 Good (0–50) | 🟡 Satisfactory (51–100) | 🟠 Moderate (101–200)
- 🔴 Poor (201–300) | 🟣 Very Poor (301–400) | ⚫ Severe (401+)

**Charts (Plotly, dark theme):**
1. Horizontal bar chart — Predicted AQI per model (color-coded by AQI category)
2. AQI gauge (best model — GradBoost)
3. Model R² performance card (from saved results CSV)

**Performance Card (loaded from CSV — no model re-training):**

| Model | City Avg R² | City Avg MAE | Rank |
|-------|------------|-------------|------|
| GradBoost | 0.9906 | 2.94 | 🥇 1 |
| RandomForest | 0.9874 | 1.64 | 🥈 2 |
| XGBoost | 0.9856 | 2.83 | 🥉 3 |
| ... | ... | ... | ... |

---

## 4. PAGE 2 — TRACK B: AQI FORECASTING

### Purpose
User selects city + provides recent 48h history → models forecast AQI at t+1h, t+6h, t+24h.

### Input Panel

| Input Field | Widget | Notes |
|-------------|--------|-------|
| City | selectbox | 18 cities |
| Horizon | multiselect | t+1h / t+6h / t+24h (default all 3) |
| Input mode | radio | Manual entry / Upload CSV |

**Manual entry mode:** 6 lag sliders (PM2.5 at t-1h, t-6h, t-24h, t-48h + rolling means)
**CSV upload mode:** accepts 48-row CSV with columns: timestamp, PM2.5, PM10, NO2, SO2, CO, Ozone, NH3, AT, RH, WS

### Output Panel

**Forecast Grid (3 × 6 table):**

```
              t+1h    t+6h    t+24h
Random Forest  145     162      178
GradBoost      148     165      181   ← 🏆
XGBoost        146     163      179
LSTM           141     158      174
BiLSTM         140     156      172
CNN-BiLSTM     130     148      163
```

**Charts:**
1. Line chart — Forecast horizon degradation (all models, 3 horizons)
2. Model comparison bar chart per horizon (3 charts in columns)
3. Confidence range band (based on test RMSE from saved results)

**Performance Panel (from saved CSV):**

| Model | t+1h R² | t+6h R² | t+24h R² |
|-------|---------|---------|----------|
| GradBoost | 0.66 | 0.44 | 0.36 | ← Best |
| RF | 0.64 | 0.43 | 0.35 | |
| LSTM | 0.40 | 0.27 | 0.19 | |
| BiLSTM | 0.41 | 0.29 | 0.21 | |
| CNN-BiLSTM | −0.15 | −0.35 | −0.71 | |

---

## 5. PAGE 3 — MODEL COMPARISON DASHBOARD

**Data source:** `outputs/final_track_a_complete.csv` + `outputs/final_track_b_complete.csv`
**No model inference on this page — pure result visualization.**

| Section | Chart Type | Content |
|---------|-----------|---------|
| Track A Ranking | Grouped bar (Plotly) | R², MAE, RMSE per model |
| Track B Ranking | Grouped bar | R² per model × horizon |
| City Heatmap | Heatmap | R² by city × model |
| Classical vs DL | Bar comparison | Avg R² classical vs DL |
| Horizon Decay | Line chart | R² decay t+1h → t+24h |
| Feature Importance | Horizontal bar | Top 20 features (GBR) |

---

## 6. PAGE 4 — ABOUT / RESEARCH PAPER

| Section | Content |
|---------|---------|
| Project overview | CPCB dataset, 18 cities, 18.7M records |
| Methodology | Dual-track framework diagram |
| Leakage audit | Summary of 11 PASS checks |
| Paper abstract | From `track_a_paper_package.md` |
| Download results | Buttons to download CSVs |
| Citation | BibTeX placeholder |

---

## 7. MODEL LOADING STRATEGY

| Model | Save format | Load method | Load time (est.) |
|-------|------------|-------------|-----------------|
| Ridge | joblib .pkl | `joblib.load()` | <0.1s |
| Random Forest | joblib .pkl | `joblib.load()` | 0.5–1s |
| GradBoost | joblib .pkl | `joblib.load()` | 0.3–0.5s |
| XGBoost | .json | `xgb.Booster.load_model()` | 0.2s |
| LSTM | .keras | `tf.keras.models.load_model()` | 1–2s |
| BiLSTM | .keras | `tf.keras.models.load_model()` | 1–2s |
| CNN-BiLSTM | .keras | `tf.keras.models.load_model()` | 1–2s |

> **Note:** Models must be saved first via `joblib.dump()` in a new export block.
> Currently only `best_delhi.keras` is saved. A `49_Export_Models` block is needed
> to serialize all 13 trained model objects before Streamlit deployment.

---

## 8. AQI CATEGORY + HEALTH ADVISORY LOGIC

```python
AQI_CATEGORIES = {
    (0,   50):  ('Good',         '#00B050', 'Air quality is satisfactory.'),
    (51,  100): ('Satisfactory',  '#92D050', 'Acceptable. Sensitive groups reduce outdoor.'),
    (101, 200): ('Moderate',      '#FFFF00', 'Sensitive groups: limit prolonged outdoor exertion.'),
    (201, 300): ('Poor',          '#FF7C00', 'Everyone: limit prolonged outdoor exertion.'),
    (301, 400): ('Very Poor',     '#FF0000', 'Avoid outdoor activities.'),
    (401, 500): ('Severe',        '#99004C', 'Stay indoors. Wear N95 mask if outside.'),
}
```

---

## 9. DEPLOYMENT REQUIREMENTS

| Requirement | Value |
|------------|-------|
| Python | 3.11 |
| Streamlit | >= 1.35 |
| TensorFlow | 2.15+ (for LSTM pages) |
| RAM (minimum) | 2 GB (classical only) / 4 GB (with LSTM) |
| Deployment target | Streamlit Cloud / Docker / Zerve App |
| Cold start time | ~8–12s (with TF model loading) |
| Inference latency | <50ms (classical) / <200ms (LSTM) |

---

## 10. NEXT STEP BEFORE BUILDING STREAMLIT

Create `49_Export_Models` block that:
1. Re-fits GBR/RF/Ridge/XGB on the full training set (no test leak)
2. Saves each to `models/city_model_{city}_{model}.joblib`
3. Saves LSTM weights per city to `models/lstm_{city}.keras`
4. Generates a model manifest JSON for the app to load at startup

> Until models are exported, Track A/B pages must use **pre-computed results CSVs**
> (show all 18-city evaluation results, not live inference).