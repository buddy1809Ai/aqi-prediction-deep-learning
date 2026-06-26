
import os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks as K_cb
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── CONFIG ────────────────────────────────────────────────────────────────────
ENG_DIR  = "outputs/engineered"
OUT_DIR  = "outputs"
Path(OUT_DIR).mkdir(exist_ok=True)

TARGET     = "AQI"
SEQ_LEN    = 24
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
RANDOM_STATE = 42

# Track-A feature candidates — same-t pollutants + met + time + interaction
SAME_T_POLLS = ["PM2.5 (µg/m³)","PM10 (µg/m³)","NO (µg/m³)","NO2 (µg/m³)",
                "NOx (ppb)","NH3 (µg/m³)","SO2 (µg/m³)","CO (mg/m³)",
                "Ozone (µg/m³)","Benzene (µg/m³)","Toluene (µg/m³)"]
MET_COLS     = ["AT (°C)","RH (%)","WS (m/s)","WD (deg)","SR (W/mt2)","BP (mmHg)"]
TIME_COLS    = ["hour","day_of_week","month","day_of_year","is_weekend","season",
                "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos"]
INTER_COLS   = ["PM25_PM10_ratio","NOx_proxy","CO_PM25_product","SO2_NO2_sum","wind_u","wind_v"]
TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

def _safe_feats(cols):
    return [c for c in cols if c != TARGET and not c.lower().startswith("aqi")]

def rmse_fn(y, yp): return float(np.sqrt(mean_squared_error(y, yp)))

# ── LSTM BUILDER ──────────────────────────────────────────────────────────────
def build_lstm(n_feats, seq_len=SEQ_LEN):
    inp = keras.Input(shape=(seq_len, n_feats))
    x   = layers.LSTM(128, return_sequences=True)(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(64)(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return mdl

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
parquets   = sorted(Path(ENG_DIR).glob("*_engineered.parquet"))
track_a_results = []

print("="*72)
print("  TRACK A — AQI ESTIMATION  (same-timestamp features → AQI(t))")
print(f"  Cities: {len(parquets)} | Target: {TARGET} | Seq_len: {SEQ_LEN}")
print("="*72)

for pq in parquets:
    city = pq.stem.replace("_engineered","")
    df   = pd.read_parquet(pq)

    if TARGET not in df.columns:
        print(f"  [{city}] ⚠ AQI column missing — skip"); continue

    # Build feature list from what's actually present
    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    feat_cols = _safe_feats(feat_cols)

    # Use only rows where ALL selected features AND target are non-null
    all_needed = feat_cols + [TARGET]
    df_model = df[all_needed].dropna(subset=all_needed)
    n = len(df_model)
    if n < 500:
        print(f"  [{city}] ⚠ only {n} clean rows — skip (missing: {df[feat_cols].isnull().mean().nlargest(5).to_dict()})")
        continue

    nt  = int(n * TRAIN_FRAC)
    nv  = int(n * VAL_FRAC)
    nte = n - nt - nv

    X_train = df_model[feat_cols].iloc[:nt].values.astype(np.float32)
    y_train = df_model[TARGET].iloc[:nt].values.astype(np.float32)
    X_val   = df_model[feat_cols].iloc[nt:nt+nv].values.astype(np.float32)
    y_val   = df_model[TARGET].iloc[nt:nt+nv].values.astype(np.float32)
    X_test  = df_model[feat_cols].iloc[nt+nv:].values.astype(np.float32)
    y_test  = df_model[TARGET].iloc[nt+nv:].values.astype(np.float32)

    sc = MinMaxScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_val_sc   = sc.transform(X_val)
    X_test_sc  = sc.transform(X_test)

    print(f"\n  [{city}]  n={n:,}  feats={len(feat_cols)}  test={nte:,}")
    print(f"  {'Model':<28} {'R²':>7} {'MAE':>8} {'RMSE':>8} {'Tr(s)':>7} {'Inf(s)':>7}")
    print("  " + "─"*67)

    # ── Classical Models ──────────────────────────────────────────────────────
    classical = [
        ("Ridge",              Ridge(alpha=1.0)),
        ("Random Forest",      RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
        ("Gradient Boosting",  GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=RANDOM_STATE)),
        ("XGBoost",            xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                                 subsample=0.8, colsample_bytree=0.8,
                                                 n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)),
    ]
    for mname, mdl in classical:
        t0 = time.time()
        mdl.fit(X_train_sc, y_train)
        tr_t = time.time() - t0
        t0 = time.time()
        yp   = mdl.predict(X_test_sc)
        inf_t= time.time() - t0
        r2v  = r2_score(y_test, yp)
        maev = mean_absolute_error(y_test, yp)
        rmsev= rmse_fn(y_test, yp)
        print(f"  {mname:<28} {r2v:>7.4f} {maev:>8.2f} {rmsev:>8.2f} {tr_t:>7.1f} {inf_t:>7.3f}")
        track_a_results.append(dict(city=city, model=mname, track="A",
                                    n_train=nt, n_test=nte, n_feats=len(feat_cols),
                                    r2=round(r2v,4), mae=round(maev,3), rmse=round(rmsev,3),
                                    train_time_s=round(tr_t,2), inference_time_s=round(inf_t,4)))

    # ── Keras LSTM ─────────────────────────────────────────────────────────────
    Xtr_s, ytr_s = make_sequences(X_train_sc, y_train, SEQ_LEN)
    Xva_s, yva_s = make_sequences(X_val_sc,   y_val,   SEQ_LEN)
    Xte_s, yte_s = make_sequences(X_test_sc,  y_test,  SEQ_LEN)

    lstm = build_lstm(len(feat_cols), SEQ_LEN)
    cb_list = [
        K_cb.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
        K_cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=0),
    ]
    t0 = time.time()
    lstm.fit(Xtr_s, ytr_s, validation_data=(Xva_s, yva_s),
             epochs=30, batch_size=256, callbacks=cb_list, verbose=0)
    tr_t = time.time() - t0
    t0 = time.time()
    yp_lstm = lstm.predict(Xte_s, verbose=0).flatten()
    inf_t = time.time() - t0
    r2v   = r2_score(yte_s, yp_lstm)
    maev  = mean_absolute_error(yte_s, yp_lstm)
    rmsev = rmse_fn(yte_s, yp_lstm)
    print(f"  {'Keras LSTM':<28} {r2v:>7.4f} {maev:>8.2f} {rmsev:>8.2f} {tr_t:>7.1f} {inf_t:>7.3f}")
    track_a_results.append(dict(city=city, model="Keras LSTM", track="A",
                                n_train=nt, n_test=len(yte_s), n_feats=len(feat_cols),
                                r2=round(r2v,4), mae=round(maev,3), rmse=round(rmsev,3),
                                train_time_s=round(tr_t,2), inference_time_s=round(inf_t,4)))
    keras.backend.clear_session()

# ── SAVE ───────────────────────────────────────────────────────────────────────
track_a_df = pd.DataFrame(track_a_results)
track_a_df.to_csv(f"{OUT_DIR}/track_a_estimation_results.csv", index=False)

# ── SUMMARY TABLE ──────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  TRACK A — SUMMARY (avg across completed cities)")
print("="*72)
_sum_a = track_a_df.groupby("model")[["r2","mae","rmse","train_time_s"]].mean().round(4)
print(_sum_a.sort_values("r2", ascending=False).to_string())
print(f"\n✓ Saved → {OUT_DIR}/track_a_estimation_results.csv  ({len(track_a_df)} rows, {track_a_df['city'].nunique()} cities)")

# ── DIAGNOSE SKIPPED CITIES ────────────────────────────────────────────────────
print("\n  SKIP DIAGNOSIS — checking missingness in SAME-T pollutants per city:")
for pq2 in parquets:
    cname = pq2.stem.replace("_engineered","")
    dft   = pd.read_parquet(pq2)
    if TARGET not in dft.columns:
        print(f"  {cname}: NO AQI COLUMN"); continue
    feat_check = [c for c in TRACK_A_CANDIDATES if c in dft.columns]
    n_complete = dft[feat_check + [TARGET]].dropna().shape[0]
    n_total    = len(dft)
    pct_miss   = 100*(1 - n_complete/n_total)
    if n_complete < 500:
        top_miss = dft[feat_check].isnull().mean().nlargest(5)
        print(f"  {cname}: n_complete={n_complete}/{n_total} ({pct_miss:.1f}% missing) | worst: {dict(top_miss.round(3))}")
    else:
        print(f"  {cname}: n_complete={n_complete}/{n_total} ✓")
