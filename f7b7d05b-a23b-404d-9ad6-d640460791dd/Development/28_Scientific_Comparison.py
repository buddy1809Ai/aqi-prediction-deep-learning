
import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

warnings.filterwarnings("ignore")

BG    = "#1D1D20"
TEXT  = "#fbfbff"
DIM   = "#909094"
PAL   = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
          "#1F77B4","#9467BD","#8C564B","#C49C94","#E377C2"]
GOLD  = "#ffd400"
GREEN = "#17b26a"
WARN  = "#f04438"

OUT = Path("outputs")
FIG = OUT / "comparison_figures"
FIG.mkdir(parents=True, exist_ok=True)
SEP = "=" * 70

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    if title:  ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=DIM, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=DIM, fontsize=9)

def load_csv_safe(path, model_name, track_label):
    """Load CSV; add model/track only if columns absent; deduplicate cols."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    # drop duplicate columns (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    # add model / track only when missing in file
    if "model" not in df.columns:
        df["model"] = model_name
    else:
        df["model"] = model_name          # always override to canonical name
    if "track" not in df.columns:
        df["track"] = track_label
    # numeric coerce row-by-row safe
    for col in ["r2","mae","rmse","train_time_s","inference_time_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ══════════════════════════════════════════════════════════════
# 1. LOAD TRACK A
# ══════════════════════════════════════════════════════════════
print(SEP); print("  LOADING TRACK A"); print(SEP)

ta_map = {
    "Ridge":       OUT/"track_a_ridge.csv",
    "RandomForest":OUT/"track_a_rf.csv",
    "GradBoost":   OUT/"track_a_gbr.csv",
    "XGBoost":     OUT/"track_a_xgb.csv",
    "LSTM":        OUT/"final_track_a_lstm.csv",
    "BiLSTM":      OUT/"track_a_bilstm.csv",
    "CNN-BiLSTM":  OUT/"track_a_cnn_bilstm.csv",
}
ta_dfs = []
for mdl, fp in ta_map.items():
    if fp.exists():
        d = load_csv_safe(fp, mdl, "A")
        ta_dfs.append(d)
        print(f"  ✓ {mdl:<15} {len(d):>3} rows")
    else:
        print(f"  ⚠ MISSING {fp.name}")

track_a_all = pd.concat(ta_dfs, ignore_index=True)
print(f"\n  Track A total rows : {len(track_a_all)}")
print(f"  Columns            : {list(track_a_all.columns)}")

# ══════════════════════════════════════════════════════════════
# 2. LOAD TRACK B
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  LOADING TRACK B"); print(SEP)

tb_map = {
    "RandomForest":OUT/"track_b_rf.csv",
    "GradBoost":   OUT/"track_b_gbr.csv",
    "XGBoost":     OUT/"track_b_xgb.csv",
    "LSTM":        OUT/"track_b_lstm.csv",
    "BiLSTM":      OUT/"track_b_bilstm.csv",
    "CNN-BiLSTM":  OUT/"track_b_cnn_bilstm.csv",
}
tb_dfs = []
for mdl, fp in tb_map.items():
    if fp.exists():
        d = load_csv_safe(fp, mdl, "B")
        tb_dfs.append(d)
        print(f"  ✓ {mdl:<15} {len(d):>3} rows")
    else:
        print(f"  ⚠ MISSING {fp.name}")

track_b_all = pd.concat(tb_dfs, ignore_index=True)
# detect horizon column
h_col = None
for c in track_b_all.columns:
    if c in ("horizon","h","step","hours"):
        h_col = c; break
if h_col:
    track_b_all[h_col] = pd.to_numeric(track_b_all[h_col], errors="coerce")
print(f"\n  Track B total rows : {len(track_b_all)}")
print(f"  Horizon column     : {h_col}")

# ══════════════════════════════════════════════════════════════
# 3. TABLE 1 — TRACK A MODEL RANKING
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  TABLE 1 — TRACK A MODEL RANKING"); print(SEP)

ta_rank = (track_a_all.groupby("model", sort=False)
           .agg(avg_r2=("r2","mean"), avg_mae=("mae","mean"),
                avg_rmse=("rmse","mean"), cities=("r2","count"))
           .reset_index().sort_values("avg_r2", ascending=False).reset_index(drop=True))
ta_rank.index = ta_rank.index + 1
print(f"\n  {'Rank':<5}{'Model':<16}{'Avg R²':>9}{'Avg MAE':>10}{'Avg RMSE':>11}{'Cities':>8}")
print("  " + "-"*60)
for i, r in ta_rank.iterrows():
    print(f"  {i:<5}{r['model']:<16}{r['avg_r2']:>9.4f}{r['avg_mae']:>10.2f}{r['avg_rmse']:>11.2f}{int(r['cities']):>8}")
ta_rank.to_csv(OUT/"track_a_model_ranking.csv", index=False)

# ══════════════════════════════════════════════════════════════
# 4. TABLE 2 — TRACK B MODEL RANKING
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  TABLE 2 — TRACK B MODEL RANKING (all horizons)"); print(SEP)

tb_rank = (track_b_all.groupby("model", sort=False)
           .agg(avg_r2=("r2","mean"), avg_mae=("mae","mean"),
                avg_rmse=("rmse","mean"), rows=("r2","count"))
           .reset_index().sort_values("avg_r2", ascending=False).reset_index(drop=True))
tb_rank.index = tb_rank.index + 1
print(f"\n  {'Rank':<5}{'Model':<16}{'Avg R²':>9}{'Avg MAE':>10}{'Avg RMSE':>11}")
print("  " + "-"*50)
for i, r in tb_rank.iterrows():
    print(f"  {i:<5}{r['model']:<16}{r['avg_r2']:>9.4f}{r['avg_mae']:>10.2f}{r['avg_rmse']:>11.2f}")
tb_rank.to_csv(OUT/"track_b_model_ranking.csv", index=False)

# ══════════════════════════════════════════════════════════════
# 5. TABLE 3 — HORIZON RANKING
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  TABLE 3 — FORECAST HORIZON RANKING"); print(SEP)
horiz_rank = pd.DataFrame()
if h_col:
    horiz_rank = (track_b_all.groupby(h_col)
                  .agg(avg_r2=("r2","mean"), avg_mae=("mae","mean"), avg_rmse=("rmse","mean"))
                  .reset_index().sort_values("avg_r2", ascending=False).reset_index(drop=True))
    horiz_rank.index = horiz_rank.index + 1
    lbl_map = {1:"t+1h", 6:"t+6h", 24:"t+24h"}
    print(f"\n  {'Rank':<5}{'Horizon':<10}{'Avg R²':>9}{'Avg MAE':>10}{'Avg RMSE':>11}")
    print("  " + "-"*48)
    for i, r in horiz_rank.iterrows():
        lbl = lbl_map.get(int(r[h_col]), str(r[h_col]))
        print(f"  {i:<5}{lbl:<10}{r['avg_r2']:>9.4f}{r['avg_mae']:>10.2f}{r['avg_rmse']:>11.2f}")
    horiz_rank.to_csv(OUT/"track_b_horizon_ranking.csv", index=False)
else:
    print("  ⚠ No horizon column — skipped")

# ══════════════════════════════════════════════════════════════
# 6. TABLE 4 — TRACK A vs B
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  TABLE 4 — TRACK A vs TRACK B"); print(SEP)
print(f"\n  {'Metric':<20}{'Track A':>14}{'Track B':>14}")
print("  " + "-"*50)
for k, la, lb in [("Avg R²",  track_a_all["r2"].mean(),   track_b_all["r2"].mean()),
                   ("Avg MAE", track_a_all["mae"].mean(),  track_b_all["mae"].mean()),
                   ("Avg RMSE",track_a_all["rmse"].mean(), track_b_all["rmse"].mean())]:
    print(f"  {k:<20}{la:>14.4f}{lb:>14.4f}")
print(f"  {'Best Model':<20}{ta_rank.iloc[0]['model']:>14}{tb_rank.iloc[0]['model']:>14}")
print(f"  {'Best Avg R²':<20}{ta_rank.iloc[0]['avg_r2']:>14.4f}{tb_rank.iloc[0]['avg_r2']:>14.4f}")

# ══════════════════════════════════════════════════════════════
# 7. TABLE 5 — DL vs CLASSICAL
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  TABLE 5 — CLASSICAL ML vs DEEP LEARNING (Track A)"); print(SEP)
DL_SET = {"LSTM","BiLSTM","CNN-BiLSTM"}
CL_SET = {"Ridge","RandomForest","GradBoost","XGBoost"}
ta_cl = track_a_all[track_a_all["model"].isin(CL_SET)]
ta_dl = track_a_all[track_a_all["model"].isin(DL_SET)]
print(f"\n  Classical ML  →  Avg R²={ta_cl['r2'].mean():.4f}  MAE={ta_cl['mae'].mean():.2f}  RMSE={ta_cl['rmse'].mean():.2f}")
print(f"  Deep Learning →  Avg R²={ta_dl['r2'].mean():.4f}  MAE={ta_dl['mae'].mean():.2f}  RMSE={ta_dl['rmse'].mean():.2f}")
dl_detail = (ta_dl.groupby("model").agg(avg_r2=("r2","mean"), avg_mae=("mae","mean"),
              avg_rmse=("rmse","mean")).reset_index().sort_values("avg_r2", ascending=False))
print("\n  DL model breakdown:")
for _, r in dl_detail.iterrows():
    print(f"    {r['model']:<16}  R²={r['avg_r2']:.4f}  MAE={r['avg_mae']:.2f}  RMSE={r['avg_rmse']:.2f}")

# ══════════════════════════════════════════════════════════════
# 8. TABLE 6 — CITY RANKING (GradBoost Track A)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}"); print("  TABLE 6 — CITY RANKING (Track A — GradBoost)"); print(SEP)
city_col = "city" if "city" in track_a_all.columns else track_a_all.columns[0]
city_rank_a = (track_a_all[track_a_all["model"]=="GradBoost"]
               .sort_values("r2", ascending=False).reset_index(drop=True))
city_rank_a.index = city_rank_a.index + 1
print(f"\n  {'Rank':<5}{'City':<25}{'R²':>9}{'MAE':>9}{'RMSE':>9}")
print("  " + "-"*60)
for i, r in city_rank_a.iterrows():
    print(f"  {i:<5}{str(r[city_col]):<25}{r['r2']:>9.4f}{r['mae']:>9.2f}{r['rmse']:>9.2f}")
city_rank_a.to_csv(OUT/"track_a_city_ranking.csv", index=False)
best_city  = city_rank_a.iloc[0][city_col]
worst_city = city_rank_a.iloc[-1][city_col]

# ══════════════════════════════════════════════════════════════
# 9. SAVE MASTER CSVs
# ══════════════════════════════════════════════════════════════
track_a_all.to_csv(OUT/"final_track_a_complete.csv", index=False)
track_b_all.to_csv(OUT/"final_track_b_complete.csv", index=False)
pd.concat([track_a_all, track_b_all], ignore_index=True).to_csv(OUT/"final_comparison.csv", index=False)
print(f"\n  ✓ final_track_a_complete.csv ({len(track_a_all)} rows)")
print(f"  ✓ final_track_b_complete.csv ({len(track_b_all)} rows)")
print(f"  ✓ final_comparison.csv")

# ══════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════

# FIG 1 — Track A model comparison (R², MAE, RMSE)
models_ord = ta_rank["model"].tolist()
fig_track_a_comparison, axes = plt.subplots(1, 3, figsize=(16, 5))
fig_track_a_comparison.patch.set_facecolor(BG)
fig_track_a_comparison.suptitle("Track A — AQI Estimation: Model Comparison",
                                  color=TEXT, fontsize=13, fontweight="bold")
for ax, col_, lbl, ci in zip(axes,
        ["avg_r2","avg_mae","avg_rmse"],
        ["Avg R²","Avg MAE","Avg RMSE"], [0,1,2]):
    vals = ta_rank[col_].tolist()
    cb   = [PAL[ci]]*len(models_ord); cb[0] = GOLD
    bars = ax.barh(models_ord[::-1], vals[::-1], color=cb[::-1], edgecolor=DIM, linewidth=0.4)
    style_ax(ax, title=lbl)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(max(bar.get_width(),0)*1.01, bar.get_y()+bar.get_height()/2,
                f"{v:.3f}" if ci==0 else f"{v:.1f}", va="center", ha="left", color=TEXT, fontsize=8)
plt.tight_layout()
plt.savefig(FIG/"fig1_track_a_model_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n  ✓ fig1_track_a_model_comparison.png")

# FIG 2 — Track B model comparison
models_b = tb_rank["model"].tolist()
fig_track_b_comparison, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig_track_b_comparison.patch.set_facecolor(BG)
fig_track_b_comparison.suptitle("Track B — AQI Forecasting: Model Comparison (all horizons)",
                                  color=TEXT, fontsize=12, fontweight="bold")
for ax, col_, lbl, ci in zip(axes2, ["avg_r2","avg_mae"], ["Avg R²","Avg MAE"], [0,1]):
    vals = tb_rank[col_].tolist()
    cb   = [PAL[ci]]*len(models_b); cb[0] = GOLD
    bars = ax.barh(models_b[::-1], vals[::-1], color=cb[::-1], edgecolor=DIM, linewidth=0.4)
    style_ax(ax, title=lbl)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(max(bar.get_width(),0)*1.01, bar.get_y()+bar.get_height()/2,
                f"{v:.3f}" if ci==0 else f"{v:.1f}", va="center", ha="left", color=TEXT, fontsize=8)
plt.tight_layout()
plt.savefig(FIG/"fig2_track_b_model_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"  ✓ fig2_track_b_model_comparison.png")

# FIG 3 — Horizon degradation curve
if h_col and len(horiz_rank) > 0:
    fig_horizon_degradation, ax3 = plt.subplots(figsize=(9, 5))
    fig_horizon_degradation.patch.set_facecolor(BG); ax3.set_facecolor(BG)
    h_vals  = sorted(track_b_all[h_col].dropna().unique())
    h_lbls  = [f"t+{int(h)}h" for h in h_vals]
    for mi, mdl in enumerate(track_b_all["model"].unique()):
        sub = track_b_all[track_b_all["model"]==mdl]
        y   = [sub[sub[h_col]==h]["r2"].mean() for h in h_vals]
        ax3.plot(h_lbls, y, marker="o", color=PAL[mi%len(PAL)], linewidth=2, markersize=6, label=mdl)
    style_ax(ax3, title="Track B — R² Degradation by Forecast Horizon",
             xlabel="Horizon", ylabel="Avg R²")
    ax3.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT, facecolor=BG)
    ax3.grid(axis="y", color=DIM, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIG/"fig3_horizon_degradation.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ fig3_horizon_degradation.png")

# FIG 4 — City × Model R² Heatmap (Track A)
models_heat = [m for m in ["Ridge","RandomForest","GradBoost","XGBoost","LSTM","BiLSTM","CNN-BiLSTM"]
               if m in track_a_all["model"].unique()]
pivot_r2 = track_a_all.pivot_table(index=city_col, columns="model",
                                    values="r2", aggfunc="mean")[models_heat]
pivot_r2 = pivot_r2.sort_values("GradBoost", ascending=False)
fig_city_heatmap, ax4 = plt.subplots(figsize=(13, 8))
fig_city_heatmap.patch.set_facecolor(BG); ax4.set_facecolor(BG)
im = ax4.imshow(pivot_r2.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.0)
ax4.set_xticks(range(len(pivot_r2.columns)))
ax4.set_xticklabels(pivot_r2.columns.tolist(), color=TEXT, fontsize=9, rotation=25, ha="right")
ax4.set_yticks(range(len(pivot_r2.index)))
ax4.set_yticklabels([str(c)[:20] for c in pivot_r2.index], color=TEXT, fontsize=8)
for sp in ax4.spines.values(): sp.set_edgecolor(DIM)
ax4.tick_params(colors=TEXT)
for i in range(len(pivot_r2.index)):
    for j in range(len(pivot_r2.columns)):
        v = pivot_r2.values[i, j]
        if not np.isnan(v):
            ax4.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v < 0.5 else "black", fontsize=7)
cbar = plt.colorbar(im, ax=ax4, fraction=0.025, pad=0.02)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)
cbar.set_label("R²", color=TEXT)
ax4.set_title("Track A — City × Model R² Heatmap", color=TEXT, fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(FIG/"fig4_city_model_heatmap.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"  ✓ fig4_city_model_heatmap.png")

# FIG 5 — Classical vs Deep Learning bar chart (Track A)
r2_by_model = track_a_all.groupby("model")["r2"].mean().sort_values(ascending=False)
fig_cl_vs_dl, ax5 = plt.subplots(figsize=(9, 5))
fig_cl_vs_dl.patch.set_facecolor(BG); ax5.set_facecolor(BG)
bar_cols = [WARN if m in DL_SET else PAL[0] for m in r2_by_model.index]
bars5 = ax5.bar(r2_by_model.index.tolist(), r2_by_model.values,
                color=bar_cols, edgecolor=DIM, linewidth=0.5)
for bar, v in zip(bars5, r2_by_model.values):
    ax5.text(bar.get_x()+bar.get_width()/2, max(v,0)+0.01, f"{v:.3f}",
             ha="center", va="bottom", color=TEXT, fontsize=8)
ax5.legend(handles=[Patch(facecolor=PAL[0], label="Classical ML"),
                    Patch(facecolor=WARN, label="Deep Learning")],
           fontsize=9, framealpha=0.3, labelcolor=TEXT, facecolor=BG)
style_ax(ax5, title="Track A — Classical ML vs Deep Learning (Avg R²)",
         xlabel="Model", ylabel="Avg R²")
ax5.set_ylim(-0.1, 1.1)
ax5.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(FIG/"fig5_classical_vs_dl.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"  ✓ fig5_classical_vs_dl.png")

# FIG 6 — LSTM vs BiLSTM vs CNN-BiLSTM per city (Track A)
dl_pivot = (track_a_all[track_a_all["model"].isin(DL_SET)]
            .pivot_table(index=city_col, columns="model", values="r2", aggfunc="mean"))
dl_cols  = [c for c in ["LSTM","BiLSTM","CNN-BiLSTM"] if c in dl_pivot.columns]
dl_pivot = dl_pivot[dl_cols].sort_values(dl_cols[0], ascending=False)
fig_dl_comparison, ax6 = plt.subplots(figsize=(12, 5))
fig_dl_comparison.patch.set_facecolor(BG); ax6.set_facecolor(BG)
x_pos = np.arange(len(dl_pivot)); width = 0.28
for ki, mdl in enumerate(dl_cols):
    ax6.bar(x_pos + ki*width, dl_pivot[mdl].values, width,
            label=mdl, color=PAL[ki+3], edgecolor=DIM, linewidth=0.4)
ax6.set_xticks(x_pos + width)
ax6.set_xticklabels([str(c)[:12] for c in dl_pivot.index],
                     rotation=40, ha="right", color=TEXT, fontsize=7)
ax6.axhline(0, color=DIM, linewidth=0.7, linestyle="--")
style_ax(ax6, title="Track A — LSTM vs BiLSTM vs CNN-BiLSTM by City",
         xlabel="City", ylabel="R²")
ax6.legend(fontsize=9, framealpha=0.3, labelcolor=TEXT, facecolor=BG)
plt.tight_layout()
plt.savefig(FIG/"fig6_dl_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"  ✓ fig6_dl_comparison.png")

# FIG 7 — Best vs Worst city (Track A)
sub_best  = track_a_all[track_a_all[city_col]==best_city].set_index("model")["r2"]
sub_worst = track_a_all[track_a_all[city_col]==worst_city].set_index("model")["r2"]
common = [m for m in models_heat if m in sub_best.index and m in sub_worst.index]
fig_best_worst, ax7 = plt.subplots(figsize=(10, 5))
fig_best_worst.patch.set_facecolor(BG); ax7.set_facecolor(BG)
xp = np.arange(len(common))
ax7.bar(xp-0.2, [sub_best.get(m, np.nan) for m in common], 0.38,
        label=f"Best: {best_city}", color=GREEN, edgecolor=DIM, linewidth=0.4)
ax7.bar(xp+0.2, [sub_worst.get(m, np.nan) for m in common], 0.38,
        label=f"Worst: {worst_city}", color=WARN, edgecolor=DIM, linewidth=0.4)
ax7.set_xticks(xp); ax7.set_xticklabels(common, color=TEXT, fontsize=9, rotation=15)
ax7.axhline(0, color=DIM, linewidth=0.7, linestyle="--")
style_ax(ax7, title="Track A — Best City vs Worst City R²", xlabel="Model", ylabel="R²")
ax7.legend(fontsize=9, framealpha=0.3, labelcolor=TEXT, facecolor=BG)
plt.tight_layout()
plt.savefig(FIG/"fig7_best_vs_worst_city.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"  ✓ fig7_best_vs_worst_city.png")

# FIG 8 — Track A vs B: R² across tracks/horizons for key models
fig_track_comparison, ax8 = plt.subplots(figsize=(11, 5))
fig_track_comparison.patch.set_facecolor(BG); ax8.set_facecolor(BG)
focus_models  = ["GradBoost","XGBoost","LSTM","BiLSTM"]
track_labels  = ["Track A"]
h_list        = sorted(track_b_all[h_col].dropna().unique()) if h_col else []
track_labels += [f"t+{int(h)}h" for h in h_list]

for mi, mdl in enumerate(focus_models):
    y_vals = [track_a_all[track_a_all["model"]==mdl]["r2"].mean()]
    for h in h_list:
        sub = track_b_all[(track_b_all["model"]==mdl) & (track_b_all[h_col]==h)]
        y_vals.append(sub["r2"].mean())
    ax8.plot(track_labels, y_vals, marker="o", color=PAL[mi], linewidth=2.2,
             markersize=7, label=mdl)

ax8.set_xticks(range(len(track_labels)))
ax8.set_xticklabels(track_labels, color=TEXT, fontsize=9)
ax8.axhline(0, color=DIM, linewidth=0.5, linestyle="--")
style_ax(ax8, title="Track A (Estimation) vs Track B (Forecasting) — R² by Model & Horizon",
         xlabel="Track / Horizon", ylabel="Avg R²")
ax8.legend(fontsize=9, framealpha=0.3, labelcolor=TEXT, facecolor=BG)
ax8.grid(axis="y", color=DIM, alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(FIG/"fig8_track_a_vs_track_b.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"  ✓ fig8_track_a_vs_track_b.png")

# ══════════════════════════════════════════════════════════════
# EXPORT SUMMARY JSON for verdict block
# ══════════════════════════════════════════════════════════════
comparison_summary = {
    "track_a_best_model":   ta_rank.iloc[0]["model"],
    "track_a_best_r2":      float(ta_rank.iloc[0]["avg_r2"]),
    "track_a_cl_avg_r2":    float(ta_cl["r2"].mean()),
    "track_a_dl_avg_r2":    float(ta_dl["r2"].mean()),
    "track_a_lstm_r2":      float(track_a_all[track_a_all["model"]=="LSTM"]["r2"].mean()),
    "track_a_bilstm_r2":    float(track_a_all[track_a_all["model"]=="BiLSTM"]["r2"].mean()),
    "track_a_cnnbilstm_r2": float(track_a_all[track_a_all["model"]=="CNN-BiLSTM"]["r2"].mean()),
    "track_b_best_model":   tb_rank.iloc[0]["model"],
    "track_b_best_r2":      float(tb_rank.iloc[0]["avg_r2"]),
    "track_b_lstm_r2":      float(track_b_all[track_b_all["model"]=="LSTM"]["r2"].mean()),
    "track_b_gbr_r2":       float(track_b_all[track_b_all["model"]=="GradBoost"]["r2"].mean()),
    "best_city_a":          str(best_city),
    "worst_city_a":         str(worst_city),
    "horizon_best":         "t+1h",
}
if h_col:
    for h in [1,6,24]:
        sub = track_b_all[track_b_all[h_col]==h]
        if len(sub): comparison_summary[f"track_b_r2_t{h}h"] = float(sub["r2"].mean())

with open(OUT/"comparison_summary.json","w") as fh:
    json.dump(comparison_summary, fh, indent=2)

print(f"\n{SEP}")
print("  FINAL SUMMARY")
print(SEP)
print(f"""
  TRACK A — AQI ESTIMATION
  ─────────────────────────────────────────────────
  Best Model   : {ta_rank.iloc[0]['model']} (Avg R²={ta_rank.iloc[0]['avg_r2']:.4f})
  2nd Model    : {ta_rank.iloc[1]['model']} (Avg R²={ta_rank.iloc[1]['avg_r2']:.4f})
  Classical ML : Avg R²={ta_cl['r2'].mean():.4f}
  Deep Learning: Avg R²={ta_dl['r2'].mean():.4f}
  Best City    : {best_city}
  Worst City   : {worst_city}

  TRACK B — AQI FORECASTING
  ─────────────────────────────────────────────────
  Best Model   : {tb_rank.iloc[0]['model']} (Avg R²={tb_rank.iloc[0]['avg_r2']:.4f})
  Best Horizon : t+1h (least degradation)
  LSTM Track B : Avg R²={track_b_all[track_b_all['model']=='LSTM']['r2'].mean():.4f}
""")
print(f"  ✓  comparison_summary.json saved")
