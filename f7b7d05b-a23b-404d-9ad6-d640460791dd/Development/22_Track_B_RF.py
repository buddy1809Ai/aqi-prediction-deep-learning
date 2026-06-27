
import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

REC_DIR  = Path('outputs/recovered')
OUT_DIR  = Path('outputs');  OUT_DIR.mkdir(exist_ok=True)
CKPT     = OUT_DIR / 'track_b_rf.csv'
TARGET   = 'AQI'
TRAIN    = 0.70
VAL      = 0.15
SEED     = 42
HORIZONS = [1, 6, 24]
SEP      = '='*70

# ── Strictly no same-t pollutants, no AQI-derived features ───────────────────
AQI_PFX      = ('AQI_lag','AQI_roll','AQI_diff','AQI_trend','AQI_cat',
                 'AQI_bucket','AQI_Index','AQI ')
SAME_T_POLLS = ['PM2.5 (µg/m³)','PM10 (µg/m³)','NO (µg/m³)','NO2 (µg/m³)',
                'NOx (ppb)','NH3 (µg/m³)','SO2 (µg/m³)','CO (mg/m³)',
                'Ozone (µg/m³)','Benzene (µg/m³)','Toluene (µg/m³)']
MET_COLS     = ['AT (°C)','BP (mmHg)','RH (%)','WS (m/s)','WD (deg)','SR (W/mt2)']
TIME_COLS    = ['hour','month','day_of_week','season','is_weekend',
                'hour_sin','hour_cos','month_sin','month_cos',
                'dow_sin','dow_cos','season_sin']
INTER_COLS   = ['wind_u','wind_v']

def rmse(y, yp):
    return float(np.sqrt(mean_squared_error(y, yp)))

# ── Load checkpoint ───────────────────────────────────────────────────────────
done_keys = set()
if CKPT.exists():
    prev      = pd.read_csv(CKPT)
    done_keys = set(zip(prev['city'], prev['horizon']))
    results   = prev.to_dict('records')
    print(f"Checkpoint: {len(prev)} rows already done")
else:
    results = []

parquets = sorted(REC_DIR.glob('*_recovered.parquet'))
print(f"\n{SEP}\n  TRACK B — RANDOM FOREST  (18 cities × 3 horizons)\n"
      f"  Features: lags + rolling + met + time\n"
      f"  EXCLUDED: same-t pollutants, all AQI-derived features\n{SEP}")

for pq in parquets:
    city = pq.stem.replace('_recovered','')
    df   = pd.read_parquet(pq)

    if TARGET not in df.columns:
        print(f"  [{city}] ✗ no AQI — skip")
        continue

    # Drop AQI-derived and same-t pollutant columns
    drop_cols = ([c for c in df.columns if c.startswith(AQI_PFX)] +
                 [c for c in SAME_T_POLLS if c in df.columns])
    df = df.drop(columns=drop_cols, errors='ignore')

    # Track-B feature set: pollutant lags + rolling + met + time
    lag_feats  = [c for c in df.columns if '_lag'  in c.lower() and not c.startswith(AQI_PFX)]
    roll_feats = [c for c in df.columns if '_roll' in c.lower() and not c.startswith(AQI_PFX)]
    met_avail  = [c for c in MET_COLS   if c in df.columns]
    time_avail = [c for c in TIME_COLS  if c in df.columns]
    inter_avail= [c for c in INTER_COLS if c in df.columns]
    feat_cols  = list(dict.fromkeys(lag_feats + roll_feats + met_avail + time_avail + inter_avail))

    if len(feat_cols) < 5:
        print(f"  [{city}] ✗ only {len(feat_cols)} features — skip")
        continue

    needed = feat_cols + [TARGET]
    df_m   = df[needed].copy()
    for c in feat_cols:
        if df_m[c].isna().any():
            df_m[c] = df_m[c].fillna(df_m[c].median())
    df_m = df_m.dropna(subset=[TARGET])

    n = len(df_m)
    if n < 500:
        print(f"  [{city}] ✗ only {n} rows — skip")
        continue

    X_base = df_m[feat_cols].values.astype(np.float32)
    y_base = df_m[TARGET].values.astype(np.float32)
    nt_base = int(n * TRAIN)
    nv_base = int(n * VAL)

    print(f"\n  [{city}]  n={n:,}  feats={len(feat_cols)}")

    for h in HORIZONS:
        if (city, h) in done_keys:
            print(f"    t+{h:02d}h  RF  ✓ skip (checkpoint)")
            continue

        # Chronological shift: predict AQI h steps ahead
        y_shifted        = np.full(n, np.nan, dtype=np.float32)
        y_shifted[:-h]   = y_base[h:]
        valid            = ~np.isnan(y_shifted)
        Xv = X_base[valid]; yv = y_shifted[valid]

        ntr = int(len(Xv) * TRAIN)
        nvl = int(len(Xv) * VAL)
        X_tr = Xv[:ntr];         y_tr = yv[:ntr]
        X_te = Xv[ntr+nvl:];     y_te = yv[ntr+nvl:]

        if len(y_te) < 100:
            print(f"    t+{h:02d}h  ✗ insufficient test rows")
            continue

        mdl = RandomForestRegressor(n_estimators=150, n_jobs=-1,
                                    max_features=0.5, random_state=SEED)
        t0 = time.time()
        mdl.fit(X_tr, y_tr)
        tr_t = time.time() - t0

        t1   = time.time()
        yp   = mdl.predict(X_te)
        inf_t= time.time() - t1

        r2v  = float(r2_score(y_te, yp))
        maev = float(mean_absolute_error(y_te, yp))
        rv   = rmse(y_te, yp)

        results.append({
            'city': city, 'horizon': h, 'model': 'RandomForest', 'track': 'B',
            'n_train': len(y_tr), 'n_test': len(y_te), 'n_feats': len(feat_cols),
            'r2': round(r2v,4), 'mae': round(maev,2), 'rmse': round(rv,2),
            'train_time_s': round(tr_t,2), 'inference_time_s': round(inf_t,4)
        })
        done_keys.add((city, h))
        pd.DataFrame(results).to_csv(CKPT, index=False)   # checkpoint
        print(f"    t+{h:02d}h  RF  R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rv:.1f}  t={tr_t:.1f}s")

track_b_rf_df = pd.DataFrame(results)
track_b_rf_df.to_csv(CKPT, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  TRACK B — RANDOM FOREST SUMMARY\n{SEP}")
for h in HORIZONS:
    sub = track_b_rf_df[track_b_rf_df['horizon'] == h]
    if len(sub) == 0: continue
    print(f"\n  Horizon t+{h:02d}h  |  cities={len(sub)}")
    print(f"    Avg R²={sub['r2'].mean():.4f}  "
          f"Avg MAE={sub['mae'].mean():.2f}  "
          f"Avg RMSE={sub['rmse'].mean():.2f}")
    print(f"    Best : {sub.loc[sub['r2'].idxmax(),'city']}  R²={sub['r2'].max():.4f}")
    print(f"    Worst: {sub.loc[sub['r2'].idxmin(),'city']}  R²={sub['r2'].min():.4f}")
print(f"\n  Saved → {CKPT}\n{SEP}")
