"""
TRACK A — LSTM — All 18 Cities (AQI Estimation)
=================================================
Architecture validated by diagnostic: 2-layer LSTM (64→32) + Dense(16) + Dense(1)
Features : same-t pollutants + met + time (NO AQI-derived cols)
Target   : AQI(t)
Saves    : outputs/final_track_a_lstm.csv
           outputs/final_track_a_complete.csv  (merge with classical)
"""
import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEP   = "=" * 72
SEQ   = 24
TRAIN = 0.70
VAL   = 0.15
SEED  = 42
tf.random.set_seed(SEED); np.random.seed(SEED)

REC_DIR = Path("outputs/recovered")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "AQI"
AQI_PFX = ("AQI_", "aqi_")

SAME_T_POLLS = [
    "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)", "NO2 (µg/m³)",
    "NOx (ppb)", "NH3 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)",
    "Ozone (µg/m³)", "Benzene (µg/m³)", "Toluene (µg/m³)",
]
MET_COLS  = ["AT (°C)", "RH (%)", "WS (m/s)", "WD (°)", "SR (W/mt2)", "BP (mmHg)"]
TIME_COLS = [
    "hour", "month", "season", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin",  "dow_cos",  "season_sin",
]
INTER_COLS = [
    "PM25_PM10_ratio", "NOx_proxy", "CO_PM25_product",
    "SO2_NO2_sum", "wind_u", "wind_v",
]
TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

# ─── helpers ────────────────────────────────────────────────────────────────
def rmse(y, yp):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yp)) ** 2)))

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def build_lstm(n_feat, seq_len):
    inp = keras.Input(shape=(seq_len, n_feat))
    x   = layers.LSTM(64, return_sequences=True,
                      dropout=0.2, recurrent_dropout=0.1)(inp)
    x   = layers.LSTM(32, return_sequences=False,
                      dropout=0.2, recurrent_dropout=0.1)(x)
    x   = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return mdl

# ─── main loop ───────────────────────────────────────────────────────────────
parquets = sorted(REC_DIR.glob("*_recovered.parquet"))
print(SEP)
print(f"  TRACK A — LSTM — All Cities ({len(parquets)} found)")
print(SEP)

lstm_results = []

for pq in parquets:
    city = pq.stem.replace("_recovered", "")
    df   = pd.read_parquet(pq)

    # drop AQI-derived cols
    drop_aqi  = [c for c in df.columns if c.startswith(AQI_PFX)]
    df        = df.drop(columns=drop_aqi, errors="ignore")

    feat_cols = [c for c in TRACK_A_CANDIDATES
                 if c in df.columns and c != TARGET]
    needed    = feat_cols + [TARGET]
    df_m      = df[needed].dropna(subset=[TARGET]).copy()

    for c in feat_cols:
        if df_m[c].isna().any():
            df_m[c] = df_m[c].fillna(df_m[c].median())

    n   = len(df_m)
    nt  = int(n * TRAIN)
    nv  = int(n * VAL)

    if n - SEQ < 500:
        print(f"  [{city}] ⚠ skipped — only {n} rows after cleaning")
        lstm_results.append({
            "city": city, "model": "LSTM",
            "R2": np.nan, "MAE": np.nan, "RMSE": np.nan,
            "train_rows": n, "n_features": len(feat_cols),
            "epochs": 0, "train_time_s": 0, "inference_time_s": 0,
            "status": "skipped",
        })
        continue

    X_all = df_m[feat_cols].values.astype(np.float32)
    y_all = df_m[TARGET].values.astype(np.float32)

    X_tr, y_tr = X_all[:nt],      y_all[:nt]
    X_va, y_va = X_all[nt:nt+nv], y_all[nt:nt+nv]
    X_te, y_te = X_all[nt+nv:],   y_all[nt+nv:]

    sc   = MinMaxScaler(); X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va); X_te_s = sc.transform(X_te)

    sc_y = MinMaxScaler()
    y_tr_s = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_s = sc_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_s = sc_y.transform(y_te.reshape(-1,1)).ravel()

    Xs_tr, ys_tr = make_sequences(X_tr_s, y_tr_s, SEQ)
    Xs_va, ys_va = make_sequences(X_va_s, y_va_s, SEQ)
    Xs_te, ys_te = make_sequences(X_te_s, y_te_s, SEQ)

    keras.backend.clear_session()
    model = build_lstm(len(feat_cols), SEQ)

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-6, verbose=0),
    ]

    t0   = time.time()
    hist = model.fit(
        Xs_tr, ys_tr,
        validation_data=(Xs_va, ys_va),
        epochs=50, batch_size=256,
        callbacks=cb_list, verbose=0,
    )
    tr_t = time.time() - t0
    ep   = len(hist.history["loss"])

    t1     = time.time()
    yp_s   = model.predict(Xs_te, verbose=0).ravel()
    inf_t  = time.time() - t1

    yp_inv = sc_y.inverse_transform(yp_s.reshape(-1,1)).ravel()
    yt_inv = sc_y.inverse_transform(ys_te.reshape(-1,1)).ravel()

    r2v  = r2_score(yt_inv, yp_inv)
    maev = mean_absolute_error(yt_inv, yp_inv)
    rmsev = rmse(yt_inv, yp_inv)

    print(f"  [{city}]  n={n:,}  feats={len(feat_cols)}  ep={ep}"
          f"  R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rmsev:.1f}  "
          f"t={tr_t:.0f}s")

    lstm_results.append({
        "city": city, "model": "LSTM",
        "R2": round(r2v, 4), "MAE": round(maev, 2), "RMSE": round(rmsev, 2),
        "train_rows": n, "n_features": len(feat_cols),
        "epochs": ep, "train_time_s": round(tr_t, 1),
        "inference_time_s": round(inf_t, 3),
        "status": "ok",
    })

    del model, Xs_tr, Xs_va, Xs_te, X_all, y_all
    import gc; gc.collect()

# ─── Save LSTM results ───────────────────────────────────────────────────────
track_a_lstm_df = pd.DataFrame(lstm_results)
track_a_lstm_df.to_csv(OUT_DIR / "final_track_a_lstm.csv", index=False)
print(f"\n  Saved → outputs/final_track_a_lstm.csv  ({len(track_a_lstm_df)} rows)")

# ─── Merge with classical results ────────────────────────────────────────────
classical_path = OUT_DIR / "final_track_a_classical.csv"
if classical_path.exists():
    classical_df = pd.read_csv(classical_path)
    # standardise column names
    classical_df = classical_df.rename(columns={
        "r2": "R2", "mae": "MAE", "rmse": "RMSE",
        "train_time": "train_time_s", "inference_time": "inference_time_s",
    })
    # keep only common columns
    shared = [c for c in track_a_lstm_df.columns if c in classical_df.columns]
    track_a_complete_df = pd.concat(
        [classical_df[shared], track_a_lstm_df[shared]], ignore_index=True
    )
    track_a_complete_df.to_csv(OUT_DIR / "final_track_a_complete.csv", index=False)
    print(f"  Saved → outputs/final_track_a_complete.csv  ({len(track_a_complete_df)} rows)")
else:
    track_a_complete_df = track_a_lstm_df.copy()
    print("  ⚠ classical CSV not found — complete file = LSTM only")

# ─── Summary table ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  TRACK A LSTM — SUMMARY")
print(SEP)
valid = track_a_lstm_df[track_a_lstm_df["status"] == "ok"]
print(f"  Cities trained : {len(valid)} / {len(parquets)}")
print(f"  Avg R²         : {valid['R2'].mean():.4f}")
print(f"  Avg MAE        : {valid['MAE'].mean():.2f}")
print(f"  Avg RMSE       : {valid['RMSE'].mean():.2f}")
best_c  = valid.loc[valid["R2"].idxmax(), "city"]
worst_c = valid.loc[valid["R2"].idxmin(), "city"]
print(f"  Best city      : {best_c}  (R²={valid['R2'].max():.4f})")
print(f"  Worst city     : {worst_c} (R²={valid['R2'].min():.4f})")
print(SEP)

# ─── FIGURE : R² bar chart ────────────────────────────────────────────────────
valid_sorted = valid.sort_values("R2", ascending=True)
fig_track_a_lstm, ax = plt.subplots(figsize=(12, 7))
fig_track_a_lstm.patch.set_facecolor("#1D1D20")
ax.set_facecolor("#1D1D20")

colors = ["#17b26a" if r >= 0.90 else "#ffd400" if r >= 0.70 else "#f04438"
          for r in valid_sorted["R2"]]
bars = ax.barh(valid_sorted["city"], valid_sorted["R2"], color=colors, edgecolor="#333")

for bar, val in zip(bars, valid_sorted["R2"]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", ha="left", color="#fbfbff", fontsize=9)

ax.axvline(0.90, color="#A1C9F4", ls="--", lw=1, alpha=0.7, label="R²=0.90")
ax.set_xlabel("R² Score", color="#909094")
ax.set_title("Track A — LSTM AQI Estimation | R² by City",
             color="#fbfbff", fontsize=13, pad=12)
ax.tick_params(colors="#909094")
ax.set_xlim(0, 1.05)
for sp in ax.spines.values():
    sp.set_edgecolor("#444")
ax.grid(True, color="#333", ls=":", axis="x")
ax.legend(facecolor="#2a2a2e", edgecolor="#444", labelcolor="#fbfbff")
fig_track_a_lstm.tight_layout()
