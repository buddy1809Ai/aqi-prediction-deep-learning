
# ============================================================
# 10_Full_Baseline_All_Cities  — Exp B Baselines for 18 Cities
# ============================================================
# Experiment B: NO AQI-derived features — honest scientific baseline
# Models: Ridge | Random Forest (n_estimators=20) | Grad. Boost (n_estimators=30)
# Reduced estimator counts to fit Fargate 5-min warm-start window
# Split: chronological 70/15/15
# ============================================================
import sys, types, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ── sklearn import stub ──────────────────────────────────────────────────────
_utils = types.ModuleType("numpy.testing._private.utils")
_utils._SUPPORTS_SVE = False
_utils.check_support_sve = lambda: False
class _SW:
    def __init__(self,*a,**k): pass
    def __call__(self,f): return f
    def __enter__(self): return self
    def __exit__(self,*a): pass
    def filter(self,*a,**k): pass
    def record(self,*a,**k): pass
_utils.suppress_warnings = _SW
for _nm in ["assert_array_equal","assert_allclose","raises","assert_warns",
            "build_err_msg","assert_array_less","assert_string_equal",
            "assert_approx_equal","measure","print_coercion_tables",
            "assert_array_max_ulp","assert_array_almost_equal_nulp",
            "assert_no_gc_cycles","TestCase","overrides",
            "_gen_alignment_data","_assert_valid_refcount"]:
    setattr(_utils, _nm, lambda *a,**k: None)
_priv = types.ModuleType("numpy.testing._private"); _priv.utils = _utils
_testing = types.ModuleType("numpy.testing"); _testing.suppress_warnings = _SW
_testing._private = _priv
for _nm in ["assert_array_equal","assert_allclose","assert_array_almost_equal",
            "assert_equal","assert_raises","assert_warns","assert_no_gc_cycles",
            "assert_array_less","assert_string_equal","assert_approx_equal",
            "TestCase","overrides"]:
    setattr(_testing, _nm, lambda *a,**k: None)
sys.modules["numpy.testing._private.utils"] = _utils
sys.modules["numpy.testing._private"]       = _priv
sys.modules["numpy.testing"]                = _testing
for _k in list(sys.modules.keys()):
    if _k.startswith("scipy") or _k.startswith("sklearn"):
        del sys.modules[_k]
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
print("sklearn loaded ✓")

# ── Config ────────────────────────────────────────────────────────────────────
BG, TEXT, DIM = "#1D1D20", "#fbfbff", "#909094"
PALETTE       = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF"]
ENG_DIR       = "outputs/engineered"
OUT           = "outputs/leakage"
TARGET        = "AQI"
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
AQI_PFX       = ("AQI_lag","AQI_roll_","AQI_diff","AQI_trend")
BASE_EXCLUDE  = {TARGET,"AQI_Category","City","city_id"}
VALID_DT      = [np.float32,np.float64,np.int8,np.int16,np.int32,np.int64]

# ── Metrics ───────────────────────────────────────────────────────────────────
def r2_np(yt,yp):
    ss_res=np.sum((yt-yp)**2); ss_tot=np.sum((yt-yt.mean())**2)
    return float(1-ss_res/(ss_tot+1e-12))
def mae_np(yt,yp): return float(np.mean(np.abs(yt-yp)))
def rmse_np(yt,yp): return float(np.sqrt(np.mean((yt-yp)**2)))

def expB_feats(df):
    return [c for c in df.columns
            if c not in BASE_EXCLUDE
            and df[c].dtype in VALID_DT
            and not df[c].isna().all()
            and not c.startswith(AQI_PFX)]

# ── City list ─────────────────────────────────────────────────────────────────
parquets   = sorted([f for f in os.listdir(ENG_DIR) if f.endswith("_engineered.parquet")])
all_cities = [f.replace("_engineered.parquet","").replace("_"," ") for f in parquets]
print(f"Cities found: {len(all_cities)}\n")

# ── Seed with pre-loaded Ahmedabad & Chennai (from 08_Baseline_Models) ────────
pre_loaded = {}
for _, row in leakage_exp_df[leakage_exp_df.experiment=="B"].iterrows():
    c = row["city"]
    pre_loaded.setdefault(c, []).append({
        "city":c,"model":row["model"],
        "n_feats":int(row["n_feats"]),
        "R2":float(row["R2"]),"MAE":float(row["MAE"]),"RMSE":float(row["RMSE"]),
        "source":"upstream"
    })

all_results = [r for rows in pre_loaded.values() for r in rows]

print("="*72)
print("  Experiment B — Full Baseline — All 18 Cities")
print("="*72)

for fname in parquets:
    city_slug = fname.replace("_engineered.parquet","")
    city      = city_slug.replace("_"," ")

    if city in pre_loaded:
        sub = [r for r in all_results if r["city"]==city]
        print(f"\n  [{city}]  (pre-loaded)")
        for r in sub:
            print(f"    {r['model']:<22} R²={r['R2']:.4f}  MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}")
        continue

    fpath = os.path.join(ENG_DIR, fname)
    df    = pd.read_parquet(fpath).sort_index()
    n     = len(df)
    nt    = int(n*TRAIN_FRAC)
    nv    = int(n*VAL_FRAC)
    ntv   = nt+nv
    y     = df[TARGET].fillna(0).values.astype(np.float32)
    fc    = expB_feats(df)
    X     = df[fc].fillna(0).astype(np.float32).values
    sc    = MinMaxScaler().fit(X[:nt])
    Xs    = sc.transform(X)
    Xtv   = Xs[:ntv]; ytv = y[:ntv]
    Xte   = Xs[ntv:]; yte = y[ntv:]

    print(f"\n  [{city}]  n={n:,}  feats={len(fc)}  test={n-ntv:,}")
    print(f"  {'Model':<22} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("  " + "─"*50)

    for mname, mdl in [
        ("Ridge", Ridge(alpha=1.0)),
        ("Random Forest", RandomForestRegressor(
            n_estimators=20, max_depth=10, min_samples_leaf=8,
            n_jobs=-1, random_state=42)),
        ("Grad. Boost", GradientBoostingRegressor(
            n_estimators=30, max_depth=4, learning_rate=0.12,
            min_samples_leaf=15, subsample=0.6, random_state=42)),
    ]:
        mdl.fit(Xtv, ytv)
        yp  = np.clip(mdl.predict(Xte), 0, 500)
        r2  = r2_np(yte, yp)
        mae = mae_np(yte, yp)
        rms = rmse_np(yte, yp)
        all_results.append(dict(
            city=city, model=mname, n_feats=len(fc),
            R2=round(r2,4), MAE=round(mae,2), RMSE=round(rms,2),
            source="computed"
        ))
        print(f"  {mname:<22} {r2:>8.4f} {mae:>8.2f} {rms:>8.2f}")

# ── Build DataFrames ─────────────────────────────────────────────────────────
baseline_all_df = pd.DataFrame(all_results).sort_values(["city","model"])

best_per_city = (
    baseline_all_df.sort_values("R2", ascending=False)
    .groupby("city").first().reset_index()
    .rename(columns={"R2":"Best_R2","MAE":"Best_MAE",
                     "RMSE":"Best_RMSE","model":"Best_Model"})
    [["city","Best_Model","Best_R2","Best_MAE","Best_RMSE","n_feats"]]
    .sort_values("Best_R2", ascending=False)
)

print("\n" + "="*72)
print("  RANKING — Best Exp-B Model per City  (True LSTM Benchmarks)")
print("="*72)
print(f"\n  {'Rk':<4}{'City':<22}{'Model':<20}{'R²':>7}{'MAE':>8}{'RMSE':>8}")
print("  " + "─"*64)
for rank, row in enumerate(best_per_city.itertuples(), 1):
    flag = "  ⚠ LOW" if row.Best_R2 < 0.80 else ""
    print(f"  {rank:<4}{row.city:<22}{row.Best_Model:<20}"
          f"{row.Best_R2:>7.4f}{row.Best_MAE:>8.2f}{row.Best_RMSE:>8.2f}{flag}")

baseline_weak_cities = best_per_city[best_per_city.Best_R2 < 0.80]["city"].tolist()
print(f"\n  ⚠ Cities below R²=0.80: {baseline_weak_cities if baseline_weak_cities else 'None'}")

# ── Save ─────────────────────────────────────────────────────────────────────
out_all  = os.path.join(OUT, "baseline_all_cities.csv")
out_best = os.path.join(OUT, "best_per_city.csv")
baseline_all_df.to_csv(out_all,  index=False)
best_per_city.to_csv(out_best, index=False)
print(f"\n  ✓ {out_all}")
print(f"  ✓ {out_best}")

# ── Chart 1: R² heatmap ──────────────────────────────────────────────────────
pivot_r2 = (
    baseline_all_df[["city","model","R2"]]
    .pivot(index="city", columns="model", values="R2")
    .sort_values("Random Forest", ascending=True)
)
MODEL_COLS = [c for c in ["Ridge","Random Forest","Grad. Boost"] if c in pivot_r2.columns]
pivot_r2   = pivot_r2[MODEL_COLS]
_data = pivot_r2.values
_rows = list(pivot_r2.index)
_cols = list(pivot_r2.columns)

fig_baseline_r2, ax = plt.subplots(figsize=(11, 9))
fig_baseline_r2.patch.set_facecolor(BG); ax.set_facecolor(BG)
im = ax.imshow(_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
for i in range(len(_rows)):
    for j in range(len(_cols)):
        v = _data[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if v > 0.45 else TEXT, fontweight="bold")
ax.set_xticks(range(len(_cols))); ax.set_xticklabels(_cols, color=TEXT, fontsize=11)
ax.set_yticks(range(len(_rows))); ax.set_yticklabels(_rows, color=TEXT, fontsize=9)
ax.tick_params(colors=TEXT)
for sp in ax.spines.values(): sp.set_visible(False)
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.ax.tick_params(colors=TEXT); cbar.set_label("R²", color=TEXT, fontsize=10)
ax.set_title("Exp B — Baseline R² (18 Cities × 3 Models)\nNo AQI-Derived Features — Honest Benchmark",
             color=TEXT, fontsize=13, fontweight="bold", pad=12)
fig_baseline_r2.tight_layout()

# ── Chart 2: Best-model R² bar ───────────────────────────────────────────────
fig_best_bar, ax2 = plt.subplots(figsize=(13, 7))
fig_best_bar.patch.set_facecolor(BG); ax2.set_facecolor(BG)
_bpc = best_per_city.sort_values("Best_R2")
_cols2 = ["#ffd400" if r < 0.80 else PALETTE[1] for r in _bpc.Best_R2]
bars2  = ax2.barh(_bpc.city, _bpc.Best_R2, color=_cols2, edgecolor="none", height=0.65)
ax2.axvline(0.80, color=PALETTE[3], lw=1.5, ls="--", label="R²=0.80 threshold")
ax2.axvline(0.90, color=PALETTE[2], lw=1.0, ls=":",  label="R²=0.90 reference")
for bar, rv in zip(bars2, _bpc.Best_R2):
    ax2.text(min(rv+0.005, 0.99), bar.get_y()+bar.get_height()/2,
             f"{rv:.3f}", va="center", color=TEXT, fontsize=8.5)
ax2.set_xlabel("R² — Exp B (no AQI-derived features)", color=TEXT, fontsize=10)
ax2.set_xlim(0, 1.08)
ax2.tick_params(colors=TEXT)
for sp in ax2.spines.values(): sp.set_visible(False)
ax2.set_title("Best Baseline R² per City — True LSTM Benchmark\n(Experiment B: Pollutants + Met + Time features only)",
              color=TEXT, fontsize=13, fontweight="bold")
ax2.legend(facecolor=BG, edgecolor=DIM, labelcolor=TEXT, fontsize=9)
fig_best_bar.tight_layout()

print("\n  fig_baseline_r2  — 18-city R² heatmap")
print("  fig_best_bar     — best model R² per city")
print("\n  ✅ Step 2 complete — full baseline all cities done.")
baseline_summary = best_per_city.copy()
