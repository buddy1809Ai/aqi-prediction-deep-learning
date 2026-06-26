
import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers

REC_DIR  = Path('outputs/recovered')
OUT_DIR  = Path('outputs');  OUT_DIR.mkdir(exist_ok=True)
CKPT     = OUT_DIR / 'track_b_lstm.csv'
TARGET   = 'AQI'
SEQ      = 24
TRAIN    = 0.70
VAL      = 0.15
SEED     = 42
HORIZONS = [1, 6, 24]
SEP      = '='*70
tf.random.set_seed(SEED); np.random.seed(SEED)

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

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def build_lstm(n_feats):
    inp = keras.Input(shape=(SEQ, n_feats))
    x   = layers.LSTM(64, return_sequences=True)(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(32)(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return mdl

# Load checkpoint
done_keys = set()
if CKPT.exists():
    prev      = pd.read_csv(CKPT)
    done_keys = set(zip(prev['city'], prev['horizon']))
    results   = prev.to_dict('records')
    print(f"Checkpoint: {len(prev)} rows done")
else:
    results = []

parquets = sorted(REC_DIR.glob('*_recovered.parquet'))
print(f"\n{SEP}\n  TRACK B — LSTM  (18 cities × 3 horizons)\n"
      f"  seq_len={SEQ}  |  Features: lags + rolling + met + time\n"
      f"  EXCLUDED: same-t pollutants, all AQI-derived features\n{SEP}")

for pq in parquets:
    city = pq.stem.replace('_recovered','')
    df   = pd.read_parquet(pq)
    if TARGET not in df.columns:
        print(f"  [{city}] ✗ no AQI — skip"); continue

    drop_cols = ([c for c in df.columns if c.startswith(AQI_PFX)] +
                 [c for c in SAME_T_POLLS if c in df.columns])
    df = df.drop(columns=drop_cols, errors='ignore')

    lag_feats  = [c for c in df.columns if '_lag'  in c.lower() and not c.startswith(AQI_PFX)]
    roll_feats = [c for c in df.columns if '_roll' in c.lower() and not c.startswith(AQI_PFX)]
    met_avail  = [c for c in MET_COLS   if c in df.columns]
    time_avail = [c for c in TIME_COLS  if c in df.columns]
    inter_avail= [c for c in INTER_COLS if c in df.columns]
    feat_cols  = list(dict.fromkeys(lag_feats + roll_feats + met_avail + time_avail + inter_avail))

    if len(feat_cols) < 5:
        print(f"  [{city}] ✗ only {len(feat_cols)} features — skip"); continue

    needed = feat_cols + [TARGET]
    df_m   = df[needed].copy()
    for c in feat_cols:
        if df_m[c].isna().any():
            df_m[c] = df_m[c].fillna(df_m[c].median())
    df_m = df_m.dropna(subset=[TARGET])
    n = len(df_m)
    if n < SEQ + 200:
        print(f"  [{city}] ✗ only {n} rows — skip"); continue

    X_base  = df_m[feat_cols].values.astype(np.float32)
    y_base  = df_m[TARGET].values.astype(np.float32)
    nt = int(n * TRAIN); nv = int(n * VAL)

    print(f"\n  [{city}]  n={n:,}  feats={len(feat_cols)}")

    for h in HORIZONS:
        if (city, h) in done_keys:
            print(f"    t+{h:02d}h  LSTM  ✓ skip"); continue

        # Chronological shift: target = AQI h steps ahead
        y_shifted       = np.full(n, np.nan, dtype=np.float32)
        y_shifted[:-h]  = y_base[h:]
        valid_mask      = ~np.isnan(y_shifted)
        Xv = X_base[valid_mask]; yv = y_shifted[valid_mask]
        ntr = int(len(Xv)*TRAIN); nvl = int(len(Xv)*VAL)

        # Scale features
        sc   = MinMaxScaler(); sc_y = MinMaxScaler()
        Xtr_s  = sc.fit_transform(Xv[:ntr])
        Xva_s  = sc.transform(Xv[ntr:ntr+nvl])
        Xte_s  = sc.transform(Xv[ntr+nvl:])
        ytr_s  = sc_y.fit_transform(yv[:ntr].reshape(-1,1)).ravel()
        yva_s  = sc_y.transform(yv[ntr:ntr+nvl].reshape(-1,1)).ravel()
        yte    = yv[ntr+nvl:]

        Xs_tr, ys_tr = make_sequences(Xtr_s, ytr_s, SEQ)
        Xs_va, ys_va = make_sequences(Xva_s, yva_s, SEQ)
        Xs_te, _     = make_sequences(Xte_s, np.zeros(len(Xte_s)), SEQ)
        yte_aligned  = yte[SEQ:]   # align with sequence output

        if len(ys_te := _) == 0 or len(Xs_te) < 50:
            print(f"    t+{h:02d}h ✗ insufficient sequences"); continue

        keras.backend.clear_session()
        model = build_lstm(len(feat_cols))
        cbs   = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=4, min_lr=1e-6, verbose=0)
        ]
        t0 = time.time()
        model.fit(Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
                  epochs=80, batch_size=256, callbacks=cbs, verbose=0)
        tr_t = time.time() - t0

        t1   = time.time()
        yp_s = model.predict(Xs_te, verbose=0).ravel()
        inf_t= time.time() - t1
        yp   = sc_y.inverse_transform(yp_s.reshape(-1,1)).ravel()

        r2v  = float(r2_score(yte_aligned, yp))
        maev = float(mean_absolute_error(yte_aligned, yp))
        rv   = rmse(yte_aligned, yp)

        results.append({'city':city,'horizon':h,'model':'LSTM','track':'B',
                         'n_train':len(ys_tr),'n_test':len(yte_aligned),'n_feats':len(feat_cols),
                         'r2':round(r2v,4),'mae':round(maev,2),'rmse':round(rv,2),
                         'train_time_s':round(tr_t,2),'inference_time_s':round(inf_t,4)})
        done_keys.add((city, h))
        pd.DataFrame(results).to_csv(CKPT, index=False)
        print(f"    t+{h:02d}h  LSTM  R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rv:.1f}  t={tr_t:.1f}s")

track_b_lstm_df = pd.DataFrame(results)
track_b_lstm_df.to_csv(CKPT, index=False)

print(f"\n{SEP}\n  TRACK B — LSTM SUMMARY\n{SEP}")
for h in HORIZONS:
    sub = track_b_lstm_df[track_b_lstm_df['horizon']==h]
    if len(sub) == 0: continue
    print(f"\n  Horizon t+{h:02d}h  |  cities={len(sub)}")
    print(f"    Avg R²={sub['r2'].mean():.4f}  Avg MAE={sub['mae'].mean():.2f}  Avg RMSE={sub['rmse'].mean():.2f}")
    print(f"    Best : {sub.loc[sub['r2'].idxmax(),'city']}  R²={sub['r2'].max():.4f}")
    print(f"    Worst: {sub.loc[sub['r2'].idxmin(),'city']}  R²={sub['r2'].min():.4f}")
print(f"\n  Saved → {CKPT}\n{SEP}")
