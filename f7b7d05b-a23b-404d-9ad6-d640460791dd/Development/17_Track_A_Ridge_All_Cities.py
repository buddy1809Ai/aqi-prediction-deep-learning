
import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
REC_DIR   = Path('outputs/recovered')
OUT_DIR   = Path('outputs');  OUT_DIR.mkdir(exist_ok=True)
CKPT      = OUT_DIR / 'track_a_ridge.csv'
TARGET    = 'AQI'
TRAIN     = 0.70
VAL       = 0.15
SEED      = 42

AQI_PFX   = ('AQI_lag','AQI_roll','AQI_diff','AQI_trend','AQI_cat','AQI_bucket','AQI ')
SAME_T_POLLS = ['PM2.5 (µg/m³)','PM10 (µg/m³)','NO (µg/m³)','NO2 (µg/m³)',
                'NOx (ppb)','NH3 (µg/m³)','SO2 (µg/m³)','CO (mg/m³)',
                'Ozone (µg/m³)','Benzene (µg/m³)','Toluene (µg/m³)']
MET_COLS  = ['AT (°C)','BP (mmHg)','RH (%)','WS (m/s)','WD (deg)','SR (W/mt2)']
TIME_COLS = ['hour','month','day_of_week','season','is_weekend',
             'hour_sin','hour_cos','month_sin','month_cos',
             'dow_sin','dow_cos','season_sin']
INTER_COLS= ['PM25_PM10_ratio','NOx_proxy','CO_PM25_product',
             'SO2_NO2_sum','wind_u','wind_v']

TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

SEP = '='*70

# ── Load checkpoint ─────────────────────────────────────────────────────────
done_cities = set()
if CKPT.exists():
    prev = pd.read_csv(CKPT)
    done_cities = set(prev['city'].unique())
    results = prev.to_dict('records')
    print(f"Checkpoint loaded — {len(done_cities)} cities already done: {done_cities}")
else:
    results = []

def rmse(y, yp): return float(np.sqrt(mean_squared_error(y, yp)))

parquets = sorted(REC_DIR.glob('*_recovered.parquet'))
print(f"\n{SEP}\n  BLOCK 1 — RIDGE — All Cities ({len(parquets)} found)\n{SEP}")

for pq in parquets:
    city = pq.stem.replace('_recovered','')
    if city in done_cities:
        print(f"  [{city}] ✓ already done — skip")
        continue

    df = pd.read_parquet(pq)
    if TARGET not in df.columns:
        print(f"  [{city}] ✗ no AQI column — skip")
        continue

    # Drop AQI-derived cols
    drop_aqi = [c for c in df.columns if c.startswith(AQI_PFX)]
    df = df.drop(columns=drop_aqi, errors='ignore')

    # Select available features
    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    needed    = feat_cols + [TARGET]
    df_m      = df[needed].copy()

    # Impute per-column median (no global dropna)
    for c in feat_cols:
        if df_m[c].isna().any():
            df_m[c] = df_m[c].fillna(df_m[c].median())
    df_m = df_m.dropna(subset=[TARGET])

    n = len(df_m)
    if n < 500:
        print(f"  [{city}] ✗ only {n} rows after impute — skip")
        continue

    nt = int(n * TRAIN)
    nv = int(n * VAL)

    X = df_m[feat_cols].values.astype(np.float32)
    y = df_m[TARGET].values.astype(np.float32)

    X_tr, y_tr = X[:nt],        y[:nt]
    X_va, y_va = X[nt:nt+nv],   y[nt:nt+nv]
    X_te, y_te = X[nt+nv:],     y[nt+nv:]

    sc = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)

    # ── Ridge ───────────────────────────────────────────────────────────────
    t0  = time.time()
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr_s, y_tr)
    tr_t = time.time() - t0

    t1 = time.time()
    yp = mdl.predict(X_te_s)
    inf_t = time.time() - t1

    r2v   = float(r2_score(y_te, yp))
    maev  = float(mean_absolute_error(y_te, yp))
    rmsev = rmse(y_te, yp)

    results.append({
        'city': city, 'model': 'Ridge', 'track': 'A',
        'n_train': nt, 'n_test': len(y_te), 'n_feats': len(feat_cols),
        'r2': round(r2v,4), 'mae': round(maev,2), 'rmse': round(rmsev,2),
        'train_time_s': round(tr_t,2), 'inference_time_s': round(inf_t,4)
    })
    # Checkpoint immediately
    pd.DataFrame(results).to_csv(CKPT, index=False)
    print(f"  [{city}]  n={n:,}  feats={len(feat_cols)}  "
          f"R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rmsev:.1f}  t={tr_t:.1f}s")

# ── Summary ──────────────────────────────────────────────────────────────────
track_a_ridge_df = pd.DataFrame(results)
track_a_ridge_df.to_csv(CKPT, index=False)

print(f"\n{SEP}")
print(f"  RIDGE SUMMARY")
print(f"  Cities done : {len(track_a_ridge_df)}")
if len(track_a_ridge_df) > 0:
    print(f"  Avg R²      : {track_a_ridge_df['r2'].mean():.4f}")
    print(f"  Avg MAE     : {track_a_ridge_df['mae'].mean():.2f}")
    print(f"  Avg RMSE    : {track_a_ridge_df['rmse'].mean():.2f}")
    print(f"  Best city   : {track_a_ridge_df.loc[track_a_ridge_df['r2'].idxmax(),'city']}  "
          f"(R²={track_a_ridge_df['r2'].max():.4f})")
    print(f"  Worst city  : {track_a_ridge_df.loc[track_a_ridge_df['r2'].idxmin(),'city']}  "
          f"(R²={track_a_ridge_df['r2'].min():.4f})")
print(f"  Saved → {CKPT}")
print(SEP)
