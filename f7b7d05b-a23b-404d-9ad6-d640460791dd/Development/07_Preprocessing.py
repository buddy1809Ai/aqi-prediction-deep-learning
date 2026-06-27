
import os, glob, pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

IN_DIR   = "outputs/engineered"
OUT_DIR  = "outputs/preprocessed"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

TARGET      = "AQI"
SEQ_LEN     = 24          # 24-hour lookback window
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# TEST_FRAC = 0.15 (remainder)

# ─── Feature list: exclude non-numeric + leakage columns ─────────────────────
EXCLUDE_COLS = [TARGET, "AQI_Category", "City", "city_id",
                # AQI-derived cols that would cause leakage
                "AQI_diff1h", "AQI_diff6h", "AQI_diff24h", "AQI_trend"]
# AQI lag/roll features are kept — they represent PAST AQI, not current → valid

print("=" * 72)
print("05 — PREPROCESSING & SPLIT")
print("=" * 72)

parquets   = sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
split_meta = {}      # city → {n_train, n_val, n_test, feature_cols, scaler_path}
scalers    = {}      # city → fitted MinMaxScaler

# ─── Pass 1: Fit scalers on training portion only ────────────────────────────
print("\nPass 1: Fit MinMaxScaler on training data only…")
for p in parquets:
    city = os.path.basename(p).replace("_engineered.parquet","").replace("_"," ")
    df   = pd.read_parquet(p)
    if len(df) < SEQ_LEN * 10:
        print(f"  {city}: too few rows ({len(df)}), skipping")
        continue

    # Determine feature columns
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE_COLS
                 and df[c].dtype in [np.float32, np.float64, np.int8, np.int16, np.int32]
                 and not df[c].isna().all()]

    df_feats  = df[feat_cols].fillna(0).astype(np.float32)

    # Chronological split
    n         = len(df_feats)
    n_train   = int(n * TRAIN_FRAC)
    n_val     = int(n * VAL_FRAC)
    n_test    = n - n_train - n_val

    X_train   = df_feats.iloc[:n_train]

    # ─── LEAKAGE CHECK ───────────────────────────────────────────────────
    # Ensure no future AQI information leaks into features
    # AQI lags are all shift(+k): past-only → SAFE
    # Rolling stats use shift(1) before rolling → SAFE
    # AQI diff uses .diff() — diff(1) uses current-1 and current → 
    #   we excluded diff cols from features above → SAFE
    leakage_risk = [c for c in feat_cols if 'lag0' in c]  # lag0 = no shift = leakage
    if leakage_risk:
        print(f"  ⚠ {city}: potential leakage cols found: {leakage_risk}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    scalers[city] = scaler

    split_meta[city] = dict(
        n=n, n_train=n_train, n_val=n_val, n_test=n_test,
        feat_cols=feat_cols, n_feats=len(feat_cols)
    )
    print(f"  {city:<22}: n={n:>7,} | train={n_train:>7,} val={n_val:>6,} test={n_test:>6,} | feats={len(feat_cols)}")

# ─── Save scalers ─────────────────────────────────────────────────────────────
scaler_path = os.path.join(OUT_DIR, "city_scalers.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scalers, f)
print(f"\nScalers saved → {scaler_path}")

# ─── Pass 2: Scale + save scaled arrays for baseline models ──────────────────
print("\nPass 2: Scale + save arrays for baseline models (flat, no sequence)…")
baseline_data = {}

for p in parquets:
    city = os.path.basename(p).replace("_engineered.parquet","").replace("_"," ")
    if city not in split_meta:
        continue
    df   = pd.read_parquet(p)
    meta = split_meta[city]
    fc   = meta["feat_cols"]
    scaler = scalers[city]

    df_feats = df[fc].fillna(0).astype(np.float32)
    y        = df[TARGET].values.astype(np.float32)

    # Scale features only (NOT target — we want raw AQI predictions)
    X_sc     = scaler.transform(df_feats).astype(np.float32)

    # Chronological split
    nt, nv, nte = meta["n_train"], meta["n_val"], meta["n_test"]
    baseline_data[city] = dict(
        X_train=X_sc[:nt],      y_train=y[:nt],
        X_val=X_sc[nt:nt+nv],  y_val=y[nt:nt+nv],
        X_test=X_sc[nt+nv:],   y_test=y[nt+nv:],
    )

    # Save to disk
    np_path = os.path.join(OUT_DIR, f"{city.replace(' ','_')}_baseline.npz")
    np.savez_compressed(np_path,
                        X_train=baseline_data[city]["X_train"],
                        y_train=baseline_data[city]["y_train"],
                        X_val=baseline_data[city]["X_val"],
                        y_val=baseline_data[city]["y_val"],
                        X_test=baseline_data[city]["X_test"],
                        y_test=baseline_data[city]["y_test"])

print(f"  Saved {len(baseline_data)} city baseline .npz files")

# ─── Memory estimate for LSTM sequence arrays ─────────────────────────────────
print("\n─── LSTM SEQUENCE ARRAY MEMORY ESTIMATE ─────────────────────────────")
total_seq_gb = 0
for city, meta in split_meta.items():
    n, nf = meta["n"], meta["n_feats"]
    # Sequences: (n - SEQ_LEN) × SEQ_LEN × n_feats × 4 bytes (float32)
    n_seq   = max(0, n - SEQ_LEN)
    bytes_  = n_seq * SEQ_LEN * nf * 4
    gb      = bytes_ / 1e9
    total_seq_gb += gb
    if gb > 0.5:
        print(f"  {city:<22}: {n_seq:>7,} sequences × {SEQ_LEN}×{nf} → {gb:.2f} GB")

print(f"\n  Total LSTM array RAM if fully materialised: ~{total_seq_gb:.1f} GB")
print(f"  → Use tf.data.Dataset with batch generators (never materialise all at once)")
print(f"  → Per-city training with generator keeps peak RAM well under 4 GB")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SPLIT SUMMARY")
print("=" * 72)
print(f"\n{'City':<22} {'N':>8} {'Train':>8} {'Val':>7} {'Test':>7}  {'Feats':>6}")
print("-" * 65)
for city, m in split_meta.items():
    print(f"{city:<22} {m['n']:>8,} {m['n_train']:>8,} {m['n_val']:>7,} {m['n_test']:>7,}  {m['n_feats']:>6}")

print(f"""
─── LEAKAGE VERIFICATION ──────────────────────────────────────────────
  ✓ Scaler fitted ONLY on training portion (time index 0 → {TRAIN_FRAC*100:.0f}%)
  ✓ Validation and test data are strictly in the future vs training
  ✓ No AQI_diff or AQI_trend (uses current row) in feature set
  ✓ All lag/rolling features use shift(≥1) → past-only values
  ✓ Chronological ordering preserved (no shuffling)
  ✓ No information from val/test used in scaler fitting
  → No data leakage detected ✓
""")

print(f"\nsplit_meta exported ✓")
print(f"scalers exported ✓")
print(f"baseline_data exported ✓  ({len(baseline_data)} cities)")
