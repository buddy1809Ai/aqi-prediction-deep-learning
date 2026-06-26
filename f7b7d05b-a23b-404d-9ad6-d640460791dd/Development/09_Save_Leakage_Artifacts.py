
# ============================================================
# 09_Save_Leakage_Artifacts — Persist all audit outputs to disk
# ============================================================
# Saves: feature_catalog, experiment results, verdict, split_meta
# All computed in 08_Baseline_Models — no recomputation needed
# Root-cause fix: ensure all Path args are str, not int
# ============================================================
import os, json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

OUT = "outputs/leakage"
os.makedirs(OUT, exist_ok=True)

# ── 1. Feature Catalog (114 features × 7 columns) ─────────────────────────
cat_path = os.path.join(OUT, "feature_catalog.csv")
feature_catalog_df.to_csv(cat_path, index=False)
print(f"✓ feature_catalog.csv  ({len(feature_catalog_df)} rows)  →  {cat_path}")

# ── 2. Leakage Experiments (Exp A/B/C/D, 2 audit cities) ─────────────────
exp_path = os.path.join(OUT, "audit_experiments.csv")
leakage_exp_df.to_csv(exp_path, index=False)
print(f"✓ audit_experiments.csv  ({len(leakage_exp_df)} rows)  →  {exp_path}")

# ── 3. Verdict dict → JSON ────────────────────────────────────────────────
def _make_serializable(obj):
    """Recursively convert numpy types to Python native."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return bool(obj)
    return obj

verdict_path = os.path.join(OUT, "verdict.json")
with open(verdict_path, "w") as fh:
    json.dump(_make_serializable(leakage_verdict), fh, indent=2)
print(f"✓ verdict.json  ({len(leakage_verdict)} keys)  →  {verdict_path}")

# ── 4. Split metadata (18 cities) ─────────────────────────────────────────
meta_path = os.path.join(OUT, "split_meta.json")
with open(meta_path, "w") as fh:
    json.dump(_make_serializable(split_meta), fh, indent=2)
print(f"✓ split_meta.json  ({len(split_meta)} cities)  →  {meta_path}")

# ── 5. Quick audit summary printout ───────────────────────────────────────
n_lk = int(feature_catalog_df["is_leaky"].sum())
n_sf = len(feature_catalog_df) - n_lk
print("\n" + "═"*60)
print("  LEAKAGE ARTIFACTS SAVED")
print("═"*60)
print(f"  Feature catalog :  {len(feature_catalog_df)} features | {n_lk} leaky | {n_sf} safe")
print(f"  Experiments     :  {len(leakage_exp_df)} rows (Exp A/B/C/D × 2 cities × 3 models)")

print("\n  Experiment Results Summary:")
print(f"  {'City':<15} {'Model':<20} {'Exp A':>7} {'Exp B':>7} {'ΔA→B':>8}")
print("  " + "─"*58)
for city in leakage_exp_df["city"].unique():
    sub = leakage_exp_df[leakage_exp_df.city == city]
    for mn in ["Ridge", "Random Forest", "Grad. Boost"]:
        rA_row = sub[(sub.experiment=="A") & (sub.model==mn)]["R2"].values
        rB_row = sub[(sub.experiment=="B") & (sub.model==mn)]["R2"].values
        if len(rA_row) == 0: continue
        rA = float(rA_row[0]); rB = float(rB_row[0]) if len(rB_row) else float("nan")
        delta = rA - rB
        verdict_sym = "🚨 SEVERE" if delta > 0.10 else "✅ MINOR"
        print(f"  {city:<15} {mn:<20} {rA:>7.4f} {rB:>7.4f} {delta:>+8.4f}  {verdict_sym}")

print("\n  Files written:")
for f in ["feature_catalog.csv", "audit_experiments.csv", "verdict.json", "split_meta.json"]:
    fp = os.path.join(OUT, f)
    sz = os.path.getsize(fp)
    print(f"    {fp}  ({sz:,} bytes)")

print("\n  ✅ Step 1 complete — all leakage artifacts saved.")
saved_leakage_paths = {
    "feature_catalog" : cat_path,
    "audit_experiments": exp_path,
    "verdict"         : verdict_path,
    "split_meta"      : meta_path,
}
