
# Block: 30_Track_A_BiLSTM
# Root cause: h5py Cython module calls platform._Processor.get() which calls
# subprocess.check_output() which Zerve intercepts via lazy_load_files.py line 28.
# The _Processor class uses a cached_property so patching after import is too late.
# Fix: patch subprocess.Popen and io.open via the _lazy_open interceptor path BEFORE
# any import of tensorflow, so the lazy file opener sees a valid path string.

import sys, os, io, subprocess

# Patch subprocess to avoid Zerve's lazy_load_files interception
_orig_popen_init = subprocess.Popen.__init__
def _patched_popen_init(self, args, **kwargs):
    # If stdout is a pipe (integer fd), don't wrap it
    if 'stdout' in kwargs and kwargs['stdout'] == subprocess.PIPE:
        pass  # fine
    _orig_popen_init(self, args, **kwargs)

# Safer: patch platform._Processor directly via cached_property reset
import platform as _plat
# Bypass the cached_property by directly setting processor on the class
class _FakeProcessor:
    @staticmethod
    def get():
        return ''
try:
    _plat._Processor = _FakeProcessor
except Exception:
    pass

# Also set the uname_result processor slot to empty string to avoid cached_property
try:
    _plat._uname_cache = None
except Exception:
    pass

import time, warnings
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

tf.random.set_seed(42); np.random.seed(42)

REC_DIR = Path('outputs/recovered')
OUT_DIR = Path('outputs');  OUT_DIR.mkdir(exist_ok=True)
CKPT    = OUT_DIR / 'track_a_bilstm.csv'
TARGET  = 'AQI'
SEQ, TRAIN, VAL = 24, 0.70, 0.15
SEP = '='*70

AQI_PFX = ('AQI_', 'aqi_')
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

def rmse(y, yp):
    return float(np.sqrt(mean_squared_error(y, yp)))

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def build_bilstm(n_feats):
    inp = keras.Input(shape=(SEQ, n_feats))
    x   = layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                           dropout=0.2, recurrent_dropout=0.1))(inp)
    x   = layers.Bidirectional(layers.LSTM(32,
                                           dropout=0.2, recurrent_dropout=0.1))(x)
    x   = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(1)(x)
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return mdl

# checkpoint
done_cities = set()
if CKPT.exists():
    prev = pd.read_csv(CKPT)
    done_cities = set(prev['city'].tolist())
    results = prev.to_dict('records')
    print(f"Checkpoint: {len(done_cities)} cities already done")
else:
    results = []

parquets = sorted(REC_DIR.glob('*_recovered.parquet'))
print(SEP); print(f"  TRACK A — BiLSTM — ({len(parquets)} cities)"); print(SEP)

for pq in parquets:
    city = pq.stem.replace('_recovered', '')
    if city in done_cities:
        print(f"  [{city}]  ✓ skip"); continue

    df = pd.read_parquet(pq)
    drop_aqi = [c for c in df.columns if c.startswith(AQI_PFX) and c != TARGET]
    df = df.drop(columns=drop_aqi, errors='ignore')

    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns and c != TARGET]
    df_m = df[feat_cols + [TARGET]].dropna(subset=[TARGET]).copy()
    for col in feat_cols:
        if df_m[col].isna().any():
            df_m[col] = df_m[col].fillna(df_m[col].median())

    c, n = city, len(df_m)
    if n - SEQ < 500:
        print(f"  [{c}]  SKIP — {n} rows"); continue

    X_all = df_m[feat_cols].values.astype(np.float32)
    y_all = df_m[TARGET].values.astype(np.float32)
    nt = int(n*TRAIN); nv = int(n*VAL)

    sc   = MinMaxScaler(); sc_y = MinMaxScaler()
    Xtr_s = sc.fit_transform(X_all[:nt])
    Xva_s = sc.transform(X_all[nt:nt+nv])
    Xte_s = sc.transform(X_all[nt+nv:])
    ytr_s = sc_y.fit_transform(y_all[:nt].reshape(-1,1)).ravel()
    yva_s = sc_y.transform(y_all[nt:nt+nv].reshape(-1,1)).ravel()
    yte_s = sc_y.transform(y_all[nt+nv:].reshape(-1,1)).ravel()

    Xs_tr, ys_tr = make_sequences(Xtr_s, ytr_s, SEQ)
    Xs_va, ys_va = make_sequences(Xva_s, yva_s, SEQ)
    Xs_te, ys_te = make_sequences(Xte_s, yte_s, SEQ)

    keras.backend.clear_session()
    model = build_bilstm(len(feat_cols))
    cbs = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                         restore_best_weights=True, verbose=0),
           keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=5, min_lr=1e-6, verbose=0)]

    t0 = time.time()
    model.fit(Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
              epochs=50, batch_size=256, callbacks=cbs, verbose=0)
    tr_t = time.time() - t0

    t1 = time.time()
    yp_s = model.predict(Xs_te, verbose=0).ravel()
    inf_t = time.time() - t1

    yp_inv = sc_y.inverse_transform(yp_s.reshape(-1,1)).ravel()
    yt_inv = sc_y.inverse_transform(ys_te.reshape(-1,1)).ravel()
    r2v = float(r2_score(yt_inv, yp_inv))
    maev = float(mean_absolute_error(yt_inv, yp_inv))
    rmsev = rmse(yt_inv, yp_inv)

    print(f"  [{c}]  n={n:,}  feats={len(feat_cols)}  "
          f"R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rmsev:.1f}  t={tr_t:.0f}s")

    results.append({'city':c,'model':'BiLSTM','track':'A',
                    'r2':round(r2v,4),'mae':round(maev,2),'rmse':round(rmsev,2),
                    'train_time_s':round(tr_t,1),'inference_time_s':round(inf_t,3),
                    'n_rows':n,'n_feats':len(feat_cols)})
    pd.DataFrame(results).to_csv(CKPT, index=False)
    del model; import gc; gc.collect()

track_a_bilstm_df = pd.DataFrame(results)
print(); print(SEP); print("  TRACK A — BiLSTM SUMMARY"); print(SEP)
vr = track_a_bilstm_df[track_a_bilstm_df['r2'] > -9]
print(f"  Cities : {len(vr)} / {len(parquets)}  |  Avg R²={vr['r2'].mean():.4f}  "
      f"Avg MAE={vr['mae'].mean():.2f}  Avg RMSE={vr['rmse'].mean():.2f}")
print(f"  Best  : {vr.loc[vr['r2'].idxmax(),'city']}  R²={vr['r2'].max():.4f}")
print(f"  Worst : {vr.loc[vr['r2'].idxmin(),'city']}  R²={vr['r2'].min():.4f}")
print(f"  Saved → {CKPT}"); print(SEP)
