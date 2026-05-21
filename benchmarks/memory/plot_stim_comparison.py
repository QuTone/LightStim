"""
LightStim vs Stim Built-in: Rotated Surface Code LER Comparison.

Two panels:
  Left:  LER vs p for both frameworks (log-log, d=3,5,7)
  Right: LER ratio (LightStim / Stim) vs p (semi-log)

Output: benchmarks/memory/results/stim_comparison.png

Usage:
    venv/bin/python benchmarks/memory/plot_stim_comparison.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

STIM_CSV = Path("benchmarks/memory/results/stim_rotated_sc.csv")
LS_CSV   = Path("benchmarks/memory/results/fig1_surface_codes.csv")
OUT      = Path("benchmarks/memory/results/stim_comparison.png")
DISTS    = [3, 5, 7]
MARKERS  = {3: "o", 5: "s", 7: "^"}
LW, MS   = 1.8, 6
FS_T, FS_L, FS_TK, FS_LEG = 10, 10, 9, 9

stim_df = pd.read_csv(STIM_CSV)
ls_df   = pd.read_csv(LS_CSV)
ls_df   = ls_df[(ls_df["code"] == "rotated_sc") & (ls_df["basis"] == "Z")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

handles_ls, handles_stim, labels = [], [], []

for d in DISTS:
    color = PALETTE_DISTANCE[d]
    marker = MARKERS[d]

    # LightStim data
    ls_sub = ls_df[ls_df["distance"] == d].sort_values("p")
    # Stim data
    st_sub = stim_df[stim_df["d"] == d].sort_values("p")

    # Merge on p
    merged = pd.merge(
        ls_sub[["p", "logical_error_rate"]].rename(columns={"logical_error_rate": "ls_ler"}),
        st_sub[["p", "logical_error_rate"]].rename(columns={"logical_error_rate": "st_ler"}),
        on="p",
    ).sort_values("p")
    if merged.empty:
        continue

    # Left panel: LER vs p
    l1, = ax1.loglog(merged["p"], merged["ls_ler"],
                     color=color, marker=marker, lw=LW, ms=MS,
                     markeredgecolor="none", linestyle="-")
    l2, = ax1.loglog(merged["p"], merged["st_ler"],
                     color=color, marker=marker, lw=LW * 0.8, ms=MS * 0.8,
                     markeredgecolor="none", linestyle="--", alpha=0.75)

    if d == DISTS[0]:
        handles_ls.append(l1)
        handles_stim.append(l2)

    # Right panel: ratio vs p
    ratio = merged["ls_ler"] / merged["st_ler"]
    ax2.semilogx(merged["p"], ratio,
                 color=color, marker=marker, lw=LW, ms=MS,
                 markeredgecolor="none", linestyle="-",
                 label=f"$d={d}$")

    labels.append(f"$d={d}$")

# Left panel styling
ax1.set_xlabel("$p$", fontsize=FS_L)
ax1.set_ylabel("LER", fontsize=FS_L)
ax1.set_title("LER vs $p$", fontsize=FS_T)
ax1.tick_params(labelsize=FS_TK)
ax1.grid(True, which="major", ls="--", alpha=0.4)
bold_ticks(ax1)

proxy_ls   = Line2D([], [], color="k", ls="-",  lw=LW,       label="LightStim")
proxy_stim = Line2D([], [], color="k", ls="--", lw=LW * 0.8, alpha=0.75, label="Stim Ref.")
dist_handles = [Line2D([], [], color=PALETTE_DISTANCE[d], marker=MARKERS[d],
                        lw=LW, ms=MS, markeredgecolor="none", label=f"$d={d}$")
                for d in DISTS]
ax1.legend(handles=[proxy_ls, proxy_stim] + dist_handles,
           fontsize=FS_LEG, frameon=True, loc="upper left")

# Right panel styling
ax2.axhline(1.0, color="k", ls=":", lw=1.0, alpha=0.6, label="Ratio = 1")
ax2.set_xlabel("$p$", fontsize=FS_L)
ax2.set_ylabel("LER$_{\,\\rm LightStim}$ / LER$_{\,\\rm Stim}$", fontsize=FS_L)
ax2.set_title("LER Ratio (LightStim / Stim)", fontsize=FS_T)
ax2.tick_params(labelsize=FS_TK)
ax2.grid(True, which="major", ls="--", alpha=0.4)
bold_ticks(ax2)
ax2.legend(fontsize=FS_LEG, frameon=True, loc="upper right")

fig.suptitle("Rotated Surface Code — LightStim vs Stim Reference",
             fontweight="bold", fontsize=FS_T + 1)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
