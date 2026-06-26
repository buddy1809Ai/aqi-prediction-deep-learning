
"""
PHASE 0 — FULL CITY FORENSICS
Reads every engineered parquet. Reports:
  • row counts (cleaned vs engineered)
  • per-feature missingness by category
  • exact drop reason per city
Saves: outputs/final_audit/city_forensics.csv
       outputs/final_audit/feature_missingness.csv
"""
import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

CLEANED_DIR   = Path("outputs/cleaned")
ENG_DIR       = Path("outputs/engineered")
OUT_DIR       = Path("outputs/final_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "AQI"

# ── Feature groups ─────────────────────────────────────────────────────────────
SAME_T_POLLS = ["PM2.5 (µg/m³)","PM10 (µg/m³)","NO (µg/m³)","NO2 (µg/m³)",
                "NOx (ppb)","NH3 (µg/m³)","SO2 (µg/m³)","CO (mg/m³)",
                "Ozone (µg/m³)","Benzene (µg/m³)","Toluene (µg/m³)"]
MET_COLS      = ["AT (°C)","RH (%)","WS (m/s)","WD (deg)","SR (W/mt2)","BP (mmHg)"]
TIME_COLS     = ["hour","day_of_week","month","day_of_year","is_weekend","season",
                 "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos"]
INTER_COLS    = ["PM25_PM10_ratio","NOx_proxy","CO_PM25_product","SO2_NO2_sum","wind_u","wind_v"]
LAG_PREFIXES  = ["PM2.5 (µg/m³)_lag","PM10 (µg/m³)_lag","NO2 (µg/m³)_lag",
                 "SO2 (µg/m³)_lag","CO (mg/m³)_lag","Ozone (µg/m³)_lag","NH3 (µg/m³)_lag"]
ROLL_PREFIXES = ["PM2.5 (µg/m³)_roll","PM10 (µg/m³)_roll"]
AQI_PREFIXES  = ["AQI_lag","AQI_roll","AQI_diff","AQI_trend"]

TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

SEP = "="*72

forensics_rows = []
miss_rows      = []

eng_parquets = sorted(ENG_DIR.glob("*_engineered.parquet"))
cln_parquets = {p.stem.replace("_cleaned",""):p for p in CLEANED_DIR.glob("*_cleaned.parquet")}

print(SEP)
print("  PHASE 0 — CITY FORENSICS")
print(f"  Engineered cities found: {len(eng_parquets)}")
print(SEP)

for pq in eng_parquets:
    city = pq.stem.replace("_engineered","")
    df   = pd.read_parquet(pq)
    n_eng = len(df)

    # cleaned row count
    cln_key = next((k for k in cln_parquets if k.lower()==city.lower()), None)
    if cln_key:
        df_cln = pd.read_parquet(cln_parquets[cln_key])
        n_cln  = len(df_cln)
    else:
        n_cln = -1

    # AQI presence
    has_aqi   = TARGET in df.columns
    n_aqi_nonnull = df[TARGET].notna().sum() if has_aqi else 0

    # Missingness per group
    def grp_miss(cols):
        present = [c for c in cols if c in df.columns]
        if not present: return 1.0
        return float(df[present].isnull().mean().mean())

    miss_same_t = grp_miss(SAME_T_POLLS)
    miss_met    = grp_miss(MET_COLS)
    miss_time   = grp_miss(TIME_COLS)
    miss_inter  = grp_miss(INTER_COLS)

    # Track-A candidates present in this file
    ta_present = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    # rows where ALL track-A present cols + AQI are non-null
    needed = ta_present + ([TARGET] if has_aqi else [])
    n_complete_strict = df[needed].dropna().shape[0] if needed else 0

    # Worst missing features
    miss_per_feat = df[ta_present].isnull().mean().nlargest(5)
    worst = {k: round(v,4) for k,v in miss_per_feat.items()}

    # Determine drop reason
    if not has_aqi:
        reason = "NO_AQI_COLUMN"
    elif n_aqi_nonnull < 100:
        reason = f"AQI_ALL_NULL ({n_aqi_nonnull} non-null)"
    elif n_complete_strict < 500:
        reason = f"DROPNA_ELIMINATES_ALL (strict n={n_complete_strict})"
    elif n_complete_strict < 2000:
        reason = f"INSUFFICIENT_ROWS (strict n={n_complete_strict})"
    else:
        reason = "OK"

    # Recoverable? i.e., if we drop >95% missing feats + impute, do we get ≥500 rows?
    # Quick estimate: rows with AQI + at least 1 core pollutant non-null
    core_poll = [c for c in ["PM2.5 (µg/m³)","PM10 (µg/m³)","NO2 (µg/m³)","SO2 (µg/m³)"] if c in df.columns]
    n_recoverable = 0
    if has_aqi and core_poll:
        mask = df[TARGET].notna() & df[core_poll].notna().any(axis=1)
        n_recoverable = int(mask.sum())
    recoverable = n_recoverable >= 500

    print(f"\n  [{city}]")
    print(f"    Cleaned rows : {n_cln:,}")
    print(f"    Engineered   : {n_eng:,}")
    print(f"    AQI non-null : {n_aqi_nonnull:,}")
    print(f"    TrackA strict: {n_complete_strict:,}")
    print(f"    Recoverable  : {n_recoverable:,} rows (est.)")
    print(f"    Reason       : {reason}")
    print(f"    MissRate     : same_t={miss_same_t:.2%}  met={miss_met:.2%}  time={miss_time:.2%}  inter={miss_inter:.2%}")
    print(f"    Worst feats  : {worst}")

    forensics_rows.append(dict(
        city=city, n_cleaned=n_cln, n_engineered=n_eng,
        n_aqi_nonnull=n_aqi_nonnull, n_tracka_strict=n_complete_strict,
        n_recoverable=n_recoverable, recoverable=recoverable,
        miss_same_t=round(miss_same_t,4), miss_met=round(miss_met,4),
        miss_time=round(miss_time,4), miss_inter=round(miss_inter,4),
        reason=reason
    ))

    # per-feature missingness
    all_groups = {
        "same_t_poll": SAME_T_POLLS,
        "meteorological": MET_COLS,
        "time": TIME_COLS,
        "interaction": INTER_COLS,
        "lag": [c for c in df.columns if any(c.startswith(p) for p in LAG_PREFIXES)],
        "rolling": [c for c in df.columns if any(c.startswith(p) for p in ROLL_PREFIXES)],
        "aqi_derived": [c for c in df.columns if any(c.lower().startswith(p) for p in AQI_PREFIXES)],
    }
    for grp, feats in all_groups.items():
        for feat in feats:
            if feat in df.columns:
                miss_rows.append(dict(city=city, group=grp, feature=feat,
                                      miss_pct=round(df[feat].isnull().mean(),4),
                                      n_nonnull=int(df[feat].notna().sum())))

# ── Save ────────────────────────────────────────────────────────────────────────
city_forensics_df = pd.DataFrame(forensics_rows)
feat_miss_df      = pd.DataFrame(miss_rows)

city_forensics_df.to_csv(OUT_DIR/"city_forensics.csv", index=False)
feat_miss_df.to_csv(OUT_DIR/"feature_missingness.csv", index=False)

print(f"\n{SEP}")
print("  FORENSICS SUMMARY")
print(SEP)
print(f"  Total cities inspected : {len(forensics_rows)}")
print(f"  Cities with reason=OK  : {(city_forensics_df.reason=='OK').sum()}")
print(f"  Recoverable cities     : {city_forensics_df.recoverable.sum()}")
print(f"  NOT recoverable        : {(~city_forensics_df.recoverable).sum()}")
print(f"\n  Drop reason breakdown:")
for r, cnt in city_forensics_df.reason.value_counts().items():
    print(f"    {r:<45} {cnt}")
print(f"\n✓ city_forensics.csv     → {OUT_DIR/'city_forensics.csv'}")
print(f"✓ feature_missingness.csv → {OUT_DIR/'feature_missingness.csv'}")
