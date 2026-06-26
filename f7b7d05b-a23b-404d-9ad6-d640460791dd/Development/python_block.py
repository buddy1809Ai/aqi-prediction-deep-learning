
import os
import glob
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_ROOT  = "CPCB_Data-20260602T165735Z-3-001/CPCB_Data"
CHUNK_SIZE = 50_000

EXPECTED_CITIES = [
    "Ahmedabad","bhopal","Chennai","Delhi NCR","GandhiNagar",
    "Hyderabad","Indore","Jaipur","Jodhpur","Mumbai",
    "Mumbai suburbs","Nagpur","Navi Mumbai","Pune",
    "Singrauli","Surat","Thane","Vapi","vishakhapattanam"
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _year(f):
    m = re.search(r'_(\d{4})_', os.path.basename(f))
    return int(m.group(1)) if m else None

def _freq(f):
    b = os.path.basename(f)
    return '1Hr' if '1Hr' in b else '15Min' if '15Min' in b else 'Unknown'

def count_rows(fpath):
    n = 0
    for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
        n += len(chunk)
    return n

def is_stub(fpath):
    """A stub file reads successfully but has 0 or 1 real data rows."""
    try:
        df = pd.read_csv(fpath, nrows=3, low_memory=False)
        return len(df) == 0
    except Exception:
        return True

# ─── STEP 1: Discover city folders ───────────────────────────────────────────
print("=" * 72)
print("PHASE 0 — DATASET INVENTORY REPORT")
print("=" * 72)

city_folders = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
)
print(f"\nCity folders found ({len(city_folders)}): {city_folders}\n")

# ─── STEP 2: File census + schema sampling ───────────────────────────────────
city_stats     = {}
schema_samples = {}

print("Scanning city folders for real vs stub files…")
for city in city_folders:
    cpath   = os.path.join(DATA_ROOT, city)
    all_csv = sorted(glob.glob(os.path.join(cpath, "*.csv")))
    
    real, stubs = [], []
    for f in all_csv:
        if is_stub(f):
            stubs.append(f)
        else:
            real.append(f)
    
    years = sorted({_year(f) for f in real if _year(f)})
    freqs = sorted({_freq(f) for f in real})

    sample_df, sample_cols, err = None, [], None
    if real:
        try:
            sample_df   = pd.read_csv(real[0], nrows=5, low_memory=False)
            sample_cols = list(sample_df.columns)
            schema_samples[city] = sample_df
        except Exception as e:
            err = str(e)

    city_stats[city] = dict(
        total=len(all_csv), real=len(real), stubs=len(stubs),
        years=years,
        year_min=min(years) if years else None,
        year_max=max(years) if years else None,
        freqs=freqs, cols=sample_cols, err=err,
        real_paths=real,
    )
    print(f"  {city:<22}: {len(real)} real, {len(stubs)} stubs | years: {years}")

# ─── STEP 3: Row counting (chunked) on real files ────────────────────────────
print("\nCounting rows per city (chunked)…")
total_rows = 0
for city in city_folders:
    rc = 0
    for f in city_stats[city]["real_paths"]:
        try:
            rc += count_rows(f)
        except Exception:
            pass
    city_stats[city]["rows"] = rc
    total_rows += rc
    print(f"  {city:<22}  {rc:>10,} rows  ({city_stats[city]['real']} files)")

# ─── STEP 4: Detect column categories ────────────────────────────────────────
POLL_KEYS = ['PM2.5','PM10','NO2','NO','NOx','SO2','CO','O3','NH3',
             'Benzene','Toluene','Xylene','RSPM','SPM']
MET_KEYS  = ['WS','WD','RH','Temp','Temperature','Humidity',
             'Wind','Rain','BP','SR','AT','Rainfall']
AQI_KEYS  = ['AQI','Air Quality Index']

city_flags = {}
for city, sdf in schema_samples.items():
    cu    = [c.strip() for c in sdf.columns]
    cu_up = [c.upper() for c in cu]
    has_aqi    = any(k.upper() in ' '.join(cu_up) for k in AQI_KEYS)
    pollutants = [p for p in POLL_KEYS if any(p.upper() in c for c in cu_up)]
    met_cols   = [m for m in MET_KEYS  if any(m.upper() in c for c in cu_up)]
    city_flags[city] = dict(has_aqi=has_aqi, pollutants=pollutants,
                            met_cols=met_cols, n_cols=len(cu), col_names=cu)

# ─── STEP 5: Global column union ─────────────────────────────────────────────
all_cols = sorted({c for sdf in schema_samples.values() for c in sdf.columns})

# ─── STEP 6: Missing / unexpected cities ─────────────────────────────────────
found_low    = {c.lower() for c in city_folders}
expected_low = {c.lower() for c in EXPECTED_CITIES}
missing_exp  = [c for c in EXPECTED_CITIES if c.lower() not in found_low]
unexpected   = [c for c in city_folders   if c.lower() not in expected_low]

# ─── PRINT FULL REPORT ───────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("DATASET INVENTORY REPORT")
print("=" * 72)

print(f"\n{'City':<22} {'Files':>6} {'Stubs':>6} {'Rows':>12} {'Yr Range':>12}  Freqs")
print("-" * 72)
for city in city_folders:
    s  = city_stats[city]
    yr = f"{s['year_min']}–{s['year_max']}" if s['year_min'] else "—"
    fq = "+".join(s['freqs']) or "?"
    print(f"{city:<22} {s['real']:>6} {s['stubs']:>6} {s['rows']:>12,}  {yr:>12}  {fq}")
print("-" * 72)
print(f"{'TOTAL':<22} "
      f"{sum(v['real'] for v in city_stats.values()):>6} "
      f"{sum(v['stubs'] for v in city_stats.values()):>6} "
      f"{total_rows:>12,}")

print("\n─── SCHEMA PER CITY ───────────────────────────────────────────────────")
for city, fl in city_flags.items():
    tag = "✓ AQI" if fl['has_aqi'] else "✗ no AQI col"
    print(f"\n  {city}  ({fl['n_cols']} cols)  [{tag}]")
    print(f"    Pollutants : {fl['pollutants']}")
    print(f"    Met cols   : {fl['met_cols']}")
    print(f"    Columns    : {fl['col_names']}")

print("\n─── CITY COVERAGE ─────────────────────────────────────────────────────")
if missing_exp:
    print(f"  ⚠ Missing: {missing_exp}")
else:
    print("  All 19 expected cities present ✓")
if unexpected:
    print(f"  Unexpected: {unexpected}")

print("\n─── YEAR GAPS ─────────────────────────────────────────────────────────")
for city in city_folders:
    yrs = city_stats[city]['years']
    if yrs and len(yrs) > 1:
        gaps = sorted(set(range(min(yrs), max(yrs)+1)) - set(yrs))
        if gaps:
            print(f"  {city}: missing years {gaps}")
    elif not yrs:
        print(f"  {city}: NO real data files")

print("\n─── GLOBAL COLUMN UNION ───────────────────────────────────────────────")
print(f"  {len(all_cols)} unique columns across dataset:")
for c in all_cols:
    print(f"    {c}")

# Memory estimate
est_gb = (total_rows * 200) / 1e9
print(f"\n─── MEMORY ESTIMATE ───────────────────────────────────────────────────")
print(f"  Total rows:           {total_rows:,}")
print(f"  Est. raw CSV memory:  ~{est_gb:.1f} GB (@ 200 bytes/row)")
print(f"  Est. parquet float32: ~{est_gb*0.15:.1f} GB (after dtype optimisation)")
print("  → Chunked reading + parquet caching is MANDATORY")

print("\n─── METHODOLOGY RECOMMENDATION ────────────────────────────────────────")
large_c = [c for c in city_folders if city_stats[c]['rows'] >= 500_000]
small_c = [c for c in city_folders if city_stats[c]['rows'] <  500_000]
print(f"  ≥500K rows: {large_c}")
print(f"  <500K rows: {small_c}")
print("""
  Recommended: GLOBAL LSTM with City Embedding
  ─────────────────────────────────────────────
  • Single LSTM captures shared temporal dynamics (monsoon, diurnal cycles)
  • City embedding encodes city-specific AQI offset (Delhi >> Jodhpur)
  • Benefits smaller cities via knowledge transfer
  • Ablation: independent LSTMs for large cities to compare
  • Cluster analysis (metro/industrial/tier-2) as supplementary experiment
""")

# Export
inventory_report = dict(
    city_stats={c: {k: v for k, v in s.items() if k != 'real_paths'}
                for c, s in city_stats.items()},
    city_flags=city_flags, total_rows=total_rows,
    n_cities=len(city_folders), missing_cities=missing_exp,
    all_cols_union=all_cols,
    schema_samples={k: v.to_dict() for k, v in schema_samples.items()},
)
print("inventory_report exported ✓")
print("=" * 72)
