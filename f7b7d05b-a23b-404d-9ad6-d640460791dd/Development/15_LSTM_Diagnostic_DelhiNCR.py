"""
LSTM DIAGNOSTIC — Delhi NCR only (Track A: AQI Estimation)
============================================================
Purpose : validate sequence generation, scaling, architecture,
          loss curves, and inverse-transform before full 18-city run.
Pass criterion : test R² > 0.70
"""
import os, time, warnings
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
SEQ   = 24          # look-back window (hours)
TRAIN = 0.70
VAL   = 0.15
SEED  = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

REC_DIR  = Path("outputs/recovered")
OUT_DIR  = Path("outputs/lstm_diagnostic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "AQI"

# ── Same-timestamp Track-A features (no AQI-derived cols) ─────────────────
SAME_T_POLLS = [
    "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)", "NO2 (µg/m³)",
    "NOx (ppb)",      "NH3 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)",
    "Ozone (µg/m³)",  "Benzene (µg/m³)", "Toluene (µg/m³)",
]
MET_COLS  = ["AT (°C)", "RH (%)", "WS (m/s)", "WD (°)", "SR (W/mt2)", "BP (mmHg)"]
TIME_COLS = [
    "hour", "month", "season", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin",  "dow_cos",  "season_sin",
]
INTER_COLS = [
    "PM25_PM10_ratio", "NOx_proxy", "CO_PM25_product",
    "SO2_NO2_sum",     "wind_u",    "wind_v",
]
AQI_PFX = ("AQI_", "aqi_")

# ─────────────────────────────────────────────────────────────────────────────
def rmse(y, yp):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yp)) ** 2)))

def make_sequences(X, y, seq_len):
    """Sliding-window sequence builder — no leakage."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
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

# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  LSTM DIAGNOSTIC — Delhi NCR")
print(SEP)

pq = REC_DIR / "Delhi_NCR_recovered.parquet"
df = pd.read_parquet(pq)
print(f"  Loaded : {len(df):,} rows  |  {df.shape[1]} columns")

# ── Build feature list from available columns ─────────────────────────────
candidates = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS
feat_cols  = [c for c in candidates if c in df.columns
              and not c.startswith(AQI_PFX)]
feat_cols  = [c for c in feat_cols if c != TARGET]

print(f"  Features available : {len(feat_cols)}")
print(f"  Target             : {TARGET}")

# ── Drop AQI-derived columns ─────────────────────────────────────────────
drop_aqi = [c for c in df.columns if c.startswith(AQI_PFX)]
df = df.drop(columns=drop_aqi, errors="ignore")

# ── Select rows with valid target ─────────────────────────────────────────
df_m = df[feat_cols + [TARGET]].dropna(subset=[TARGET])

# Impute remaining NaNs (median per column)
for c in feat_cols:
    if df_m[c].isna().any():
        df_m[c] = df_m[c].fillna(df_m[c].median())

n  = len(df_m)
nt = int(n * TRAIN)
nv = int(n * VAL)
nte = n - nt - nv

print(f"  Total rows  : {n:,}  |  Train {nt:,}  Val {nv:,}  Test {nte:,}")

X_all = df_m[feat_cols].values.astype(np.float32)
y_all = df_m[TARGET].values.astype(np.float32)

# ── Chronological split ───────────────────────────────────────────────────
X_tr, y_tr = X_all[:nt],      y_all[:nt]
X_va, y_va = X_all[nt:nt+nv], y_all[nt:nt+nv]
X_te, y_te = X_all[nt+nv:],   y_all[nt+nv:]

# ── Scale on TRAIN only ───────────────────────────────────────────────────
sc = MinMaxScaler()
X_tr_s = sc.fit_transform(X_tr)
X_va_s = sc.transform(X_va)
X_te_s = sc.transform(X_te)

# Scale target separately for LSTM
sc_y = MinMaxScaler()
y_tr_s = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
y_va_s = sc_y.transform(y_va.reshape(-1, 1)).ravel()

# ── Sequence generation ───────────────────────────────────────────────────
Xs_tr, ys_tr = make_sequences(X_tr_s, y_tr_s, SEQ)
Xs_va, ys_va = make_sequences(X_va_s, y_va_s, SEQ)
Xs_te, ys_te = make_sequences(X_te_s,
                               sc_y.transform(y_te.reshape(-1,1)).ravel(),
                               SEQ)

print(f"\n  Sequence shape  — Train : {Xs_tr.shape}  |  Val : {Xs_va.shape}  |  Test : {Xs_te.shape}")
print(f"  Target shape    — Train : {ys_tr.shape}  |  Val : {ys_va.shape}  |  Test : {ys_te.shape}")

# ── Leakage check: target column must NOT appear in feature matrix ────────
assert TARGET not in feat_cols, "TARGET LEAKAGE DETECTED — AQI in features!"
print(f"  Leakage check   : PASS (AQI not in features)")

# ── Build LSTM ────────────────────────────────────────────────────────────
model = build_lstm(len(feat_cols), SEQ)
model.summary()

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=10,
                            restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=5, min_lr=1e-6, verbose=0),
    callbacks.ModelCheckpoint(str(OUT_DIR / "best_delhi.keras"),
                              monitor="val_loss", save_best_only=True,
                              verbose=0),
]

print(f"\n  Training LSTM (up to 50 epochs, EarlyStopping patience=10)…")
t0   = time.time()
hist = model.fit(
    Xs_tr, ys_tr,
    validation_data=(Xs_va, ys_va),
    epochs=50,
    batch_size=256,
    callbacks=cb_list,
    verbose=0,
)
train_time = time.time() - t0
epochs_ran = len(hist.history["loss"])
print(f"  Training done   : {epochs_ran} epochs  |  {train_time:.1f}s")

# ── Evaluate on TEST ──────────────────────────────────────────────────────
yp_s   = model.predict(Xs_te, verbose=0).ravel()
yp_inv = sc_y.inverse_transform(yp_s.reshape(-1, 1)).ravel()
yt_inv = sc_y.inverse_transform(ys_te.reshape(-1, 1)).ravel()

r2_test  = r2_score(yt_inv, yp_inv)
mae_test = mean_absolute_error(yt_inv, yp_inv)
rmse_test = rmse(yt_inv, yp_inv)

# Train-set quick eval (on sequences)
yp_tr_s   = model.predict(Xs_tr, verbose=0).ravel()
yp_tr_inv = sc_y.inverse_transform(yp_tr_s.reshape(-1,1)).ravel()
yt_tr_inv = sc_y.inverse_transform(ys_tr.reshape(-1,1)).ravel()
r2_train  = r2_score(yt_tr_inv, yp_tr_inv)

print(f"\n  ── RESULTS ──────────────────────────────────────────────────")
print(f"  Train R²   : {r2_train:.4f}")
print(f"  Test  R²   : {r2_test:.4f}")
print(f"  Test  MAE  : {mae_test:.2f}")
print(f"  Test  RMSE : {rmse_test:.2f}")

# ── Diagnostic verdict ────────────────────────────────────────────────────
pass_threshold = 0.70
if r2_test >= pass_threshold:
    diag_verdict = f"PASS ✅  (R²={r2_test:.4f} ≥ {pass_threshold})"
    proceed_lstm = True
else:
    diag_verdict = f"FAIL ❌  (R²={r2_test:.4f} < {pass_threshold})"
    proceed_lstm = False

print(f"\n  DIAGNOSTIC VERDICT : {diag_verdict}")
print(f"  Proceed to full 18-city LSTM : {proceed_lstm}")
print(SEP)

# ─── FIGURE 1 : Training & Validation Loss Curves ────────────────────────
fig_lstm_loss, ax = plt.subplots(figsize=(10, 5))
fig_lstm_loss.patch.set_facecolor("#1D1D20")
ax.set_facecolor("#1D1D20")

epochs_x = range(1, epochs_ran + 1)
ax.plot(epochs_x, hist.history["loss"],     color="#A1C9F4", lw=2, label="Train Loss")
ax.plot(epochs_x, hist.history["val_loss"], color="#FFB482", lw=2, label="Val Loss",   ls="--")
ax.set_xlabel("Epoch", color="#909094")
ax.set_ylabel("MSE Loss", color="#909094")
ax.set_title("LSTM Diagnostic — Training & Validation Loss\nDelhi NCR | Track A (AQI Estimation)",
             color="#fbfbff", fontsize=13, pad=12)
ax.legend(facecolor="#2a2a2e", edgecolor="#444", labelcolor="#fbfbff")
ax.tick_params(colors="#909094")
for sp in ax.spines.values():
    sp.set_edgecolor("#444")
ax.grid(True, color="#333", ls=":")
fig_lstm_loss.tight_layout()

# ─── FIGURE 2 : Actual vs Predicted ──────────────────────────────────────
n_plot = min(1000, len(yt_inv))
fig_lstm_pred, ax2 = plt.subplots(figsize=(12, 5))
fig_lstm_pred.patch.set_facecolor("#1D1D20")
ax2.set_facecolor("#1D1D20")

ax2.plot(range(n_plot), yt_inv[:n_plot],  color="#A1C9F4", lw=1.2, alpha=0.9, label="Actual AQI")
ax2.plot(range(n_plot), yp_inv[:n_plot],  color="#FFB482", lw=1.2, alpha=0.9, label="Predicted AQI", ls="--")
ax2.set_xlabel("Test Sample Index", color="#909094")
ax2.set_ylabel("AQI", color="#909094")
ax2.set_title(f"LSTM Diagnostic — Actual vs Predicted AQI (first {n_plot} test samples)\n"
              f"Delhi NCR  |  Test R²={r2_test:.4f}  MAE={mae_test:.1f}  RMSE={rmse_test:.1f}",
              color="#fbfbff", fontsize=12, pad=12)
ax2.legend(facecolor="#2a2a2e", edgecolor="#444", labelcolor="#fbfbff")
ax2.tick_params(colors="#909094")
for sp in ax2.spines.values():
    sp.set_edgecolor("#444")
ax2.grid(True, color="#333", ls=":")
fig_lstm_pred.tight_layout()

# ── Export diagnostic dict ────────────────────────────────────────────────
lstm_diagnostic = {
    "city"          : "Delhi_NCR",
    "input_shape"   : list(Xs_tr.shape[1:]),
    "output_shape"  : [1],
    "n_features"    : len(feat_cols),
    "seq_len"       : SEQ,
    "epochs_ran"    : epochs_ran,
    "train_time_s"  : round(train_time, 1),
    "r2_train"      : round(r2_train,  4),
    "r2_test"       : round(r2_test,   4),
    "mae_test"      : round(mae_test,  2),
    "rmse_test"     : round(rmse_test, 2),
    "verdict"       : diag_verdict,
    "proceed"       : proceed_lstm,
}

import json
with open(OUT_DIR / "lstm_diagnostic.json", "w") as fh:
    json.dump(lstm_diagnostic, fh, indent=2)
print(f"\n  Saved → {OUT_DIR}/lstm_diagnostic.json")
print(f"  Saved → {OUT_DIR}/best_delhi.keras")
