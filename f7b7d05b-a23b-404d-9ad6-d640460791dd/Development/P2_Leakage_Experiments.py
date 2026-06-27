
"""
PHASE 2 — LEAKAGE EXPERIMENTS (Exp-A / B / C)
For every recovered city run Random Forest with three feature sets:
  Exp-A: same-t pollutants + weather + time  (estimation — may reconstruct AQI formula)
  Exp-B: weather + time ONLY                 (pure meteorological signal)
  Exp-C: pollutant LAGS + rolling + weather + time  (true forecasting signal)
Quantifies: formula-reconstruction signal vs true predictive signal.
Saves: outputs/final_audit/leakage_experiments.csv
"""
import os, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

REC_DIR = Path("outputs/recovered")
OUT_DIR = Path("outputs/final_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "AQI"
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
RANDOM_STATE = 42
AQI_PFXS     = ["aqi_lag","aqi_roll","aqi_diff","aqi_trend"]

SAME_T_POLLS = ["PM2.5 (µg/m³)","PM10 (µg/m³)","NO (µg/m³)","NO2 (µg/m³)",
                "NOx (ppb)","NH3 (µg/m³)","SO2 (µg/m³)","CO (mg/m³)",
                "Ozone (µg/m³)","Benzene (µg/m³)","Toluene (µg/m³)"]
MET_COLS     = ["AT (°C)","RH (%)","WS (m/s)","WD (deg)","SR (W/mt2)","BP (mmHg)"]
TIME_COLS    = ["hour","day_of_week","month","day_of_year","is_weekend","season",
                "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos"]
INTER_COLS   = ["PM25_PM10_ratio","NOx_proxy","CO_PM25_product","SO2_NO2_sum","wind_u","wind_v"]

def rmse(y, yp):  return float(np.sqrt(mean_squared_error(y, yp)))

SEP = "="*72
results = []
parquets = sorted(REC_DIR.glob("*_recovered.parquet"))

print(SEP)
print("  PHASE 2 — LEAKAGE EXPERIMENTS (RandomForest, 100 trees)")
print("  Exp-A: same-t pollutants + met + time")
print("  Exp-B: met + time ONLY")
print("  Exp-C: pollutant lags + rolling + met + time")
print(SEP)

for pq in parquets:
    city = pq.stem.replace("_recovered","")
    df   = pd.read_parquet(pq)

    if TARGET not in df.columns: continue
    df = df.dropna(subset=[TARGET])
    n  = len(df)
    if n < 500: continue

    nt  = int(n * TRAIN_FRAC)
    nv  = int(n * VAL_FRAC)

    def split_and_train(feats):
        feats = [f for f in feats if f in df.columns and f != TARGET
                 and not any(f.lower().startswith(p) for p in AQI_PFXS)]
        if len(feats) < 2: return None, None, None, feats
        X = df[feats].values.astype(np.float32)
        y = df[TARGET].values.astype(np.float32)
        Xtr, ytr = X[:nt], y[:nt]
        Xte, yte = X[nt+nv:], y[nt+nv:]
        if len(Xte) < 50: return None, None, None, feats
        rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xte)
        return r2_score(yte, yp), mean_absolute_error(yte, yp), rmse(yte, yp), feats

    # Exp-A: same-t pollutants + met + time
    exp_a_feats = SAME_T_POLLS + INTER_COLS + MET_COLS + TIME_COLS
    r2_a, mae_a, rmse_a, fa = split_and_train(exp_a_feats)

    # Exp-B: met + time only
    exp_b_feats = MET_COLS + TIME_COLS
    r2_b, mae_b, rmse_b, fb = split_and_train(exp_b_feats)

    # Exp-C: lags + rolling + met + time (true forecasting signal, no same-t polls)
    lag_feats  = [c for c in df.columns if "_lag" in c.lower()
                  and not any(c.lower().startswith(p) for p in AQI_PFXS)]
    roll_feats = [c for c in df.columns if "_roll" in c.lower()
                  and not any(c.lower().startswith(p) for p in AQI_PFXS)]
    exp_c_feats = lag_feats + roll_feats + MET_COLS + TIME_COLS
    r2_c, mae_c, rmse_c, fc = split_and_train(exp_c_feats)

    delta_ab = (r2_a - r2_b) if (r2_a is not None and r2_b is not None) else None
    delta_ac = (r2_a - r2_c) if (r2_a is not None and r2_c is not None) else None

    print(f"\n  [{city}]  n={n:,}  train={nt:,}  test={n-nt-nv:,}")
    print(f"    Exp-A (same-t pollutants): R²={r2_a:.4f}  MAE={mae_a:.2f}  RMSE={rmse_a:.2f}  feats={len(fa)}")
    print(f"    Exp-B (met+time only)    : R²={r2_b:.4f}  MAE={mae_b:.2f}  RMSE={rmse_b:.2f}  feats={len(fb)}")
    print(f"    Exp-C (lags+roll+met)    : R²={r2_c:.4f}  MAE={mae_c:.2f}  RMSE={rmse_c:.2f}  feats={len(fc)}")
    print(f"    ΔA-B (formula signal)    : {delta_ab:+.4f}")
    print(f"    ΔA-C (same-t advantage)  : {delta_ac:+.4f}")

    for exp_id, r2v, maev, rmsev, nf in [
        ("Exp-A_sameT",   r2_a, mae_a, rmse_a, len(fa)),
        ("Exp-B_metOnly", r2_b, mae_b, rmse_b, len(fb)),
        ("Exp-C_lags",    r2_c, mae_c, rmse_c, len(fc)),
    ]:
        results.append(dict(city=city, experiment=exp_id, n_train=nt, n_test=n-nt-nv,
                            n_feats=nf, r2=round(r2v,4) if r2v else None,
                            mae=round(maev,3) if maev else None,
                            rmse=round(rmsev,3) if rmsev else None))

leakage_exp_df = pd.DataFrame(results)
leakage_exp_df.to_csv(OUT_DIR/"leakage_experiments.csv", index=False)

# Summary
print(f"\n{SEP}")
print("  LEAKAGE EXPERIMENT SUMMARY — Average R² across cities")
print(SEP)
_summ = leakage_exp_df.groupby("experiment")["r2"].agg(["mean","std","min","max"]).round(4)
print(_summ.to_string())
print(f"\n  Interpretation:")
print(f"   Exp-A >> Exp-B → AQI formula is being reconstructed from pollutants")
print(f"   Exp-C ≈ Exp-A  → Lags capture most predictive signal (true forecasting works)")
print(f"   Exp-B low      → Weather/time alone insufficient → pollutants essential")
print(f"\n✓ Saved → {OUT_DIR/'leakage_experiments.csv'}")
