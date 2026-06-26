
"""
PHASE 1 — CITY RECOVERY
For every engineered parquet:
  1. Drop features with >95% missingness IN THIS CITY
  2. Impute remaining numerics: median → ffill → bfill
  3. Drop AQI-derived features (leakage prevention)
  4. Build per-city valid feature set (whatever is available)
  5. Report row counts before / after recovery
Saves recovered parquets to outputs/recovered/
"""
import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ENG_DIR  = Path("outputs/engineered")
OUT_DIR  = Path("outputs/recovered")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "AQI"
MISS_THRESH  = 0.95   # drop feature if >95% missing IN THIS CITY
MIN_ROWS     = 500    # minimum usable rows after recovery

# Features that are ALWAYS excluded (AQI-derived → leakage)
AQI_DERIVED_PREFIXES = ["aqi_lag","aqi_roll","aqi_diff","aqi_trend"]

# Feature groups (safe — no AQI information)
SAME_T_POLLS = ["PM2.5 (µg/m³)","PM10 (µg/m³)","NO (µg/m³)","NO2 (µg/m³)",
                "NOx (ppb)","NH3 (µg/m³)","SO2 (µg/m³)","CO (mg/m³)",
                "Ozone (µg/m³)","Benzene (µg/m³)","Toluene (µg/m³)"]
MET_COLS     = ["AT (°C)","RH (%)","WS (m/s)","WD (deg)","SR (W/mt2)","BP (mmHg)"]
TIME_COLS    = ["hour","day_of_week","month","day_of_year","is_weekend","season",
                "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos"]
INTER_COLS   = ["PM25_PM10_ratio","NOx_proxy","CO_PM25_product","SO2_NO2_sum","wind_u","wind_v"]

SEP = "="*72

recovery_log = []
eng_parquets = sorted(ENG_DIR.glob("*_engineered.parquet"))

print(SEP)
print("  PHASE 1 — CITY RECOVERY")
print(f"  Strategy: drop >95%-missing features, then median/ffill/bfill imputation")
print(f"  AQI-derived features: always excluded")
print(SEP)

for pq in eng_parquets:
    city = pq.stem.replace("_engineered","")
    df   = pd.read_parquet(pq).copy()
    n0   = len(df)

    # ── Step 0: require AQI ───────────────────────────────────────────────────
    if TARGET not in df.columns or df[TARGET].isna().all():
        print(f"\n  [{city}] ✗ No AQI column — UNRECOVERABLE"); continue

    # ── Step 1: drop AQI-derived features (leakage prevention) ────────────────
    aqi_derived = [c for c in df.columns
                   if any(c.lower().startswith(p) for p in AQI_DERIVED_PREFIXES)]
    df = df.drop(columns=aqi_derived, errors="ignore")

    # ── Step 2: drop >95% missing features (per-city) ─────────────────────────
    miss_rate = df.drop(columns=[TARGET]).isnull().mean()
    drop_high_miss = miss_rate[miss_rate > MISS_THRESH].index.tolist()
    df = df.drop(columns=drop_high_miss, errors="ignore")
    n_dropped_feats = len(drop_high_miss) + len(aqi_derived)

    # ── Step 3: impute remaining numeric columns ────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_no_target = [c for c in num_cols if c != TARGET]

    # Median imputation (fit on first 70% = train portion, apply to all)
    n_train = int(len(df) * 0.70)
    medians = df[num_cols_no_target].iloc[:n_train].median()
    df[num_cols_no_target] = df[num_cols_no_target].fillna(medians)

    # Forward fill then backward fill for remaining
    df[num_cols_no_target] = df[num_cols_no_target].ffill().bfill()

    # ── Step 4: drop rows where AQI itself is null ──────────────────────────────
    n_before_aqi_drop = len(df)
    df = df.dropna(subset=[TARGET])
    n1 = len(df)

    # ── Step 5: classify final feature set ─────────────────────────────────────
    remaining_cols = df.columns.tolist()
    final_same_t   = [c for c in SAME_T_POLLS if c in remaining_cols]
    final_met      = [c for c in MET_COLS     if c in remaining_cols]
    final_time     = [c for c in TIME_COLS    if c in remaining_cols]
    final_inter    = [c for c in INTER_COLS   if c in remaining_cols]

    # Track-A features (same-t estimation)
    track_a_feats = [c for c in final_same_t + final_met + final_time + final_inter
                     if c in remaining_cols and c != TARGET]

    # Track-B features (forecasting — lags + rolling + met + time only)
    lag_feats  = [c for c in remaining_cols if "_lag" in c.lower() and
                  not any(c.lower().startswith(p) for p in AQI_DERIVED_PREFIXES)]
    roll_feats = [c for c in remaining_cols if "_roll" in c.lower() and
                  not any(c.lower().startswith(p) for p in AQI_DERIVED_PREFIXES)]
    track_b_feats = [c for c in lag_feats + roll_feats + final_met + final_time
                     if c != TARGET]

    status = "✓ RECOVERED" if n1 >= MIN_ROWS else "✗ UNRECOVERABLE"

    # ── Step 6: save recovered parquet ─────────────────────────────────────────
    if n1 >= MIN_ROWS:
        out_path = OUT_DIR / f"{city}_recovered.parquet"
        df.to_parquet(out_path, index=True)

    print(f"\n  [{city}] {status}")
    print(f"    Rows  : {n0:,} → after recovery: {n1:,}  ({n0-n1:,} lost)")
    print(f"    Feats dropped (high-miss): {len(drop_high_miss)}  |  AQI-derived: {len(aqi_derived)}")
    print(f"    Track-A feats: {len(track_a_feats)}  |  Track-B feats: {len(track_b_feats)}")
    print(f"    Same-T polls : {len(final_same_t)}/{len(SAME_T_POLLS)}  |  Met: {len(final_met)}/{len(MET_COLS)}")

    recovery_log.append(dict(
        city=city, n_raw=n0, n_recovered=n1,
        n_dropped_feats=n_dropped_feats, n_aqi_derived_removed=len(aqi_derived),
        track_a_feats=len(track_a_feats), track_b_feats=len(track_b_feats),
        same_t_polls=len(final_same_t), met_cols=len(final_met),
        time_cols=len(final_time), status=status.strip("✓ ✗ ")
    ))

# ── Summary ──────────────────────────────────────────────────────────────────
recovery_df = pd.DataFrame(recovery_log)
recovery_df.to_csv("outputs/final_audit/city_feature_recovery.csv", index=False)

n_ok = (recovery_df.n_recovered >= MIN_ROWS).sum()
print(f"\n{SEP}")
print(f"  RECOVERY SUMMARY")
print(f"  Cities recovered (≥{MIN_ROWS} rows) : {n_ok} / {len(recovery_log)}")
print(f"  Cities unrecoverable               : {len(recovery_log)-n_ok}")
print(SEP)
print(recovery_df[["city","n_raw","n_recovered","track_a_feats","track_b_feats","status"]].to_string(index=False))
print(f"\n✓ Saved → outputs/final_audit/city_feature_recovery.csv")
print(f"✓ Saved recovered parquets → outputs/recovered/")
