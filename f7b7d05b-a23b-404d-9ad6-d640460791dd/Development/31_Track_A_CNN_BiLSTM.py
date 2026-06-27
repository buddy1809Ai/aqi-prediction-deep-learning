
import os, time, warnings
import _io as _c_io          # C-level io — never monkey-patched by Zerve
import io as _io_mod

# ── Fix: Zerve patches io.open and rejects integer file-descriptors (fds).
# h5py (pulled in by TF) calls platform._Processor → subprocess.check_output
# → subprocess.Popen → io.open(c2pread_fd_int, 'rb', bufsize).  Zerve's
# wrapper raises TypeError because c2pread is an int, not a path.
#
# The C-level _io.open (imported above as _c_io) handles both paths AND int
# fds natively and is never touched by Zerve.  We store it before patching
# and install a shim on io.open that routes int fds to _c_io.open directly.
#
# os.fdopen cannot be used as a fallback because in Python 3.11 (frozen os)
# it internally calls io.open — which would recurse through our shim.

_zerve_io_open  = _io_mod.open           # Zerve's patched path-only opener
_real_open      = _c_io.open             # C-level, handles ints + paths

def _fd_safe_io_open(file, mode='r', *args, **kwargs):
    if isinstance(file, int):
        return _real_open(file, mode, *args, **kwargs)   # int fd → C-level
    return _zerve_io_open(file, mode, *args, **kwargs)   # path → Zerve open

_io_mod.open = _fd_safe_io_open   # subprocess.Popen goes through io.open

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks as kcb

_io_mod.open = _zerve_io_open     # restore for normal file I/O below

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

SEP  = "=" * 70
SEQ  = 24
TRAIN, VAL = 0.70, 0.15
REC_DIR = Path("outputs/recovered")
OUT_DIR = Path("outputs");  OUT_DIR.mkdir(exist_ok=True)
CKPT    = OUT_DIR / "track_a_cnn_bilstm.csv"
TARGET  = "AQI"

AQI_PFX      = ("AQI_", "aqi_")
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["AT","RH","WS","WD","SR","BP"]
TIME_COLS    = ["hour","month","day_of_week","season","is_weekend",
                "hour_sin","hour_cos","month_sin","month_cos",
                "dow_sin","dow_cos","season_sin"]
INTER_COLS   = ["PM25_PM10_ratio","NOx_proxy","CO_PM25_product",
                "SO2_NO2_sum","wind_u","wind_v"]
TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b))**2)))

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def build_cnn_bilstm(n_feats):
    inp = keras.Input(shape=(SEQ, n_feats))
    x   = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inp)
    x   = layers.MaxPooling1D(pool_size=2)(x)
    x   = layers.Bidirectional(layers.LSTM(64))(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return mdl

# ── Checkpoint resume ─────────────────────────────────────────────────────────
done_cities = set()
if CKPT.exists():
    prev = pd.read_csv(CKPT)
    done_cities = set(prev["city"].tolist())
    results = prev.to_dict("records")
    print(f"Checkpoint: {len(done_cities)} cities already done")
else:
    results = []

parquets = sorted(REC_DIR.glob("*_recovered.parquet"))
print(SEP)
print(f"  TRACK A — CNN-BiLSTM — All Cities ({len(parquets)} found)")
print(SEP)

cb_list_base = [
    kcb.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    kcb.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
]

for pq in parquets:
    city = pq.stem.replace("_recovered", "")
    if city in done_cities:
        print(f"  [{city}]  ✓ skip")
        continue

    df = pd.read_parquet(pq)
    drop_aqi = [c for c in df.columns if c.startswith(AQI_PFX) and c != TARGET]
    df = df.drop(columns=drop_aqi, errors="ignore")

    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    needed    = feat_cols + [TARGET]
    df_m      = df[needed].dropna()
    c, n      = city, len(df_m)

    if n < SEQ + 200:
        print(f"  [{c}]  SKIP — only {n} rows after dropna")
        continue

    X_all = df_m[feat_cols].values.astype(np.float32)
    y_all = df_m[TARGET].values.astype(np.float32)

    nt = int(n * TRAIN);  nv = int(n * VAL)
    X_tr, y_tr = X_all[:nt],      y_all[:nt]
    X_va, y_va = X_all[nt:nt+nv], y_all[nt:nt+nv]
    X_te, y_te = X_all[nt+nv:],   y_all[nt+nv:]

    sc   = MinMaxScaler().fit(X_tr)
    sc_y = MinMaxScaler().fit(y_tr.reshape(-1,1))

    Xtr_s = sc.transform(X_tr);  Xva_s = sc.transform(X_va);  Xte_s = sc.transform(X_te)
    ytr_s = sc_y.transform(y_tr.reshape(-1,1)).ravel()
    yva_s = sc_y.transform(y_va.reshape(-1,1)).ravel()

    Xs_tr, ys_tr = make_sequences(Xtr_s, ytr_s, SEQ)
    Xs_va, ys_va = make_sequences(Xva_s, yva_s, SEQ)
    Xs_te, ys_te = make_sequences(Xte_s,
                                  sc_y.transform(y_te.reshape(-1,1)).ravel(), SEQ)

    keras.backend.clear_session()
    model = build_cnn_bilstm(len(feat_cols))
    t0 = time.time()
    model.fit(Xs_tr, ys_tr,
              validation_data=(Xs_va, ys_va),
              epochs=100, batch_size=256,
              callbacks=cb_list_base, verbose=0)
    tr_t = time.time() - t0

    t1 = time.time()
    yp_s   = model.predict(Xs_te, verbose=0).ravel()
    inf_t  = time.time() - t1
    yp_inv = sc_y.inverse_transform(yp_s.reshape(-1,1)).ravel()
    yt_inv = sc_y.inverse_transform(ys_te.reshape(-1,1)).ravel()

    r2v   = float(r2_score(yt_inv, yp_inv))
    maev  = float(mean_absolute_error(yt_inv, yp_inv))
    rmsev = rmse(yt_inv, yp_inv)

    print(f"  [{c}]  n={n:,}  feats={len(feat_cols)}  "
          f"R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rmsev:.1f}  t={tr_t:.0f}s")

    results.append({"city":c,"model":"CNN-BiLSTM","track":"A",
                    "r2":round(r2v,4),"mae":round(maev,2),"rmse":round(rmsev,2),
                    "train_time_s":round(tr_t,1),"inference_time_s":round(inf_t,3),
                    "n_rows":n,"n_feats":len(feat_cols)})
    pd.DataFrame(results).to_csv(CKPT, index=False)
    del model; import gc; gc.collect()

track_a_cnn_bilstm_df = pd.DataFrame(results)

print()
print(SEP)
print("  TRACK A — CNN-BiLSTM SUMMARY")
print(SEP)
valid_r2 = track_a_cnn_bilstm_df[track_a_cnn_bilstm_df["r2"] > -9]
print(f"  Cities trained : {len(valid_r2)} / {len(parquets)}")
print(f"  Avg R²         : {valid_r2['r2'].mean():.4f}")
print(f"  Avg MAE        : {valid_r2['mae'].mean():.2f}")
print(f"  Avg RMSE       : {valid_r2['rmse'].mean():.2f}")
print(f"  Best city      : {valid_r2.loc[valid_r2['r2'].idxmax(),'city']}  "
      f"(R²={valid_r2['r2'].max():.4f})")
print(f"  Worst city     : {valid_r2.loc[valid_r2['r2'].idxmin(),'city']}  "
      f"(R²={valid_r2['r2'].min():.4f})")
print(f"  Saved → {CKPT}")
print(SEP)
