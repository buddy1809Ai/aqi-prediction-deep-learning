
# ============================================================
# 12 — ENVIRONMENT VERIFICATION
# Test every required package one-by-one; PASS / FAIL
# ============================================================

results = []

# numpy
try:
    import numpy as _np
    results.append(("numpy",        _np.__version__,          "PASS"))
except Exception as _e:
    results.append(("numpy",        str(_e)[:60],              "FAIL"))

# pandas
try:
    import pandas as _pd
    results.append(("pandas",       _pd.__version__,          "PASS"))
except Exception as _e:
    results.append(("pandas",       str(_e)[:60],              "FAIL"))

# scikit-learn
try:
    import sklearn as _sk
    results.append(("scikit-learn", _sk.__version__,          "PASS"))
except Exception as _e:
    results.append(("scikit-learn", str(_e)[:60],              "FAIL"))

# xgboost
try:
    import xgboost as _xgb
    results.append(("xgboost",      _xgb.__version__,         "PASS"))
except Exception as _e:
    results.append(("xgboost",      str(_e)[:60],              "FAIL"))

# h5py
try:
    import h5py as _h5
    results.append(("h5py",         _h5.__version__,          "PASS"))
except Exception as _e:
    results.append(("h5py",         str(_e)[:60],              "FAIL"))

# tensorflow
try:
    import tensorflow as _tf
    results.append(("tensorflow",   _tf.__version__,          "PASS"))
except Exception as _e:
    results.append(("tensorflow",   str(_e)[:80],              "FAIL"))

# keras
try:
    import keras as _keras
    results.append(("keras",        _keras.__version__,       "PASS"))
except Exception as _e:
    results.append(("keras",        str(_e)[:80],              "FAIL"))

# matplotlib
try:
    import matplotlib as _mpl
    results.append(("matplotlib",   _mpl.__version__,         "PASS"))
except Exception as _e:
    results.append(("matplotlib",   str(_e)[:60],              "FAIL"))

# scipy
try:
    import scipy as _sp
    results.append(("scipy",        _sp.__version__,          "PASS"))
except Exception as _e:
    results.append(("scipy",        str(_e)[:60],              "FAIL"))

# pyarrow
try:
    import pyarrow as _pa
    results.append(("pyarrow",      _pa.__version__,          "PASS"))
except Exception as _e:
    results.append(("pyarrow",      str(_e)[:60],              "FAIL"))

# ── Print table ──────────────────────────────────────────────
SEP = "=" * 66
print(SEP)
print("  ENVIRONMENT VERIFICATION — PACKAGE REPORT")
print(SEP)
print(f"  {'Package':<18}  {'Version / Error':<38}  Status")
print("-" * 66)
for _pkg, _ver, _st in results:
    _icon = "PASS ✅" if _st == "PASS" else "FAIL ❌"
    print(f"  {_pkg:<18}  {_ver:<38}  {_icon}")
print(SEP)

# ── Capability flags ─────────────────────────────────────────
_passed     = {r[0] for r in results if r[2] == "PASS"}
_tf_ok      = "tensorflow"   in _passed
_sk_ok      = "scikit-learn" in _passed
_xgb_ok     = "xgboost"      in _passed
_np_ok      = "numpy"        in _passed
_pd_ok      = "pandas"       in _passed

print()
print(SEP)
print("  CAPABILITY READINESS SUMMARY")
print(SEP)

_cap = [
    ("Track A — AQI Estimation  (Ridge/RF/GBR)",            _sk_ok,       "sklearn present"              if _sk_ok  else "sklearn MISSING"),
    ("Track A — AQI Estimation  (MLP deep-learning proxy)",  _sk_ok,       "MLPRegressor via sklearn"     if _sk_ok  else "sklearn MISSING"),
    ("Track B — AQI Forecasting (RF/GBR)",                   _sk_ok,       "sklearn present"              if _sk_ok  else "sklearn MISSING"),
    ("Track B — AQI Forecasting (MLP deep-learning proxy)",  _sk_ok,       "MLPRegressor via sklearn"     if _sk_ok  else "sklearn MISSING"),
    ("LSTM Training — Keras/TensorFlow",                     _tf_ok,       "TensorFlow READY"             if _tf_ok  else "NOT available — MLP proxy used"),
    ("XGBoost baseline",                                     _xgb_ok,      "xgboost present"              if _xgb_ok else "NOT installed — GBR used"),
    ("Parquet I/O (pyarrow)",                                "pyarrow" in _passed, "pyarrow present" if "pyarrow" in _passed else "MISSING"),
]

for _task, _ready, _note in _cap:
    _sym = "✅" if _ready else "⚠️ "
    print(f"  {_sym}  {_task}")
    print(f"       → {_note}")
print(SEP)

print()
print("  OVERALL VERDICT:")
if _tf_ok:
    print("  ✅ TensorFlow AVAILABLE — full Keras LSTM pipeline cleared.")
else:
    print("  ❌ TensorFlow NOT available.")
    if _sk_ok:
        print("  ⚠️  Fallback: sklearn MLPRegressor as deep-learning proxy.")
        print("     Ridge / RF / GBR / MLP — fully valid scientific pipeline.")

print()
_dl_backend = "keras" if _tf_ok else ("mlp" if _sk_ok else "none")
print(f"  dl_backend = '{_dl_backend}'")

env_status = {
    "numpy": _np_ok, "pandas": _pd_ok, "sklearn": _sk_ok,
    "xgboost": _xgb_ok, "tensorflow": _tf_ok,
    "track_a_ready": _sk_ok, "track_b_ready": _sk_ok,
    "lstm_keras": _tf_ok, "lstm_fallback": _sk_ok,
    "dl_backend": _dl_backend,
}
print("  env_status exported ✓")
print(SEP)
