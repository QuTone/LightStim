"""
Bell ZZ-LS teleportation — Absolute LER vs routing distance.

X-axis: Routing distance (2×, 4×, 8×)
Y-axis: LER  (absolute, log scale)
Lines:  Teleport |Z⟩ (solid) and |X⟩ (dashed), color-coded by d = 3, 5, 7
Fixed:  p = 1e-3

Output: results/bell_routing_overhead_absolute.png

Usage:
    venv/bin/python eval/logical_circuit_benchmark/bell-teleportation/plot_bell_routing_absolute.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from src.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

DIST_CSV = Path("eval/logical_circuit_benchmark/bell-teleportation/results/ls_zz_dist_results.csv")
BASE_CSV = Path("eval/logical_circuit_benchmark/bell-teleportation/results/ls_zz_results.csv")
OUT      = Path("eval/logical_circuit_benchmark/bell-teleportation/results/bell_routing_overhead_absolute.png")

DISTS   = [3, 5, 7]
MULTS   = [1, 2, 4, 8]
P_FIX   = 1e-3

MARKERS  = {3: "o", 5: "s", 7: "^"}
LSTYLES  = {"Z": "-", "X": "--"}
STATE_LABELS = {"Z": r"Teleport $|Z\rangle$", "X": r"Teleport $|X\rangle$"}

LW, MS = 2.0, 7
FS_TITLE  = 11
FS_LABEL  = 11
FS_TICK   = 10
FS_LEGEND = 9

# ── Load data ─────────────────────────────────────────────────────────────────
df_dist = pd.read_csv(DIST_CSV)
df_base = pd.read_csv(BASE_CSV)
df_base['routing_mult'] = 1

df = pd.concat([
    df_base[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
    df_dist[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
], ignore_index=True)
df_p = df[np.isclose(df['p'], P_FIX)]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)

for d in DISTS:
    color  = PALETTE_DISTANCE[d]
    marker = MARKERS[d]
    for state in ["Z", "X"]:
        ls = LSTYLES[state]
        sub = df_p[(df_p['d'] == d) & (df_p['state'] == state)]
        lers = []
        for mult in MULTS:
            row = sub[sub['routing_mult'] == mult]
            lers.append(row['logical_error_rate'].values[0] if not row.empty else np.nan)
        ax.semilogy(MULTS, lers,
                    color=color, marker=marker, lw=LW, ms=MS,
                    markeredgecolor='none', linestyle=ls)

ax.set_xticks(MULTS)
ax.set_xticklabels([f'{m}×' for m in MULTS])
ax.set_xlabel('Routing distance', fontsize=FS_LABEL)
ax.set_ylabel('LER', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, which='both', ls='--', alpha=0.4)
bold_ticks(ax)

# Legend: distance (color) + state (linestyle)
dist_handles = [
    Line2D([], [], color=PALETTE_DISTANCE[d], marker=MARKERS[d],
           lw=LW, ms=MS, markeredgecolor='none', label=f'$d={d}$')
    for d in DISTS
]
state_handles = [
    Line2D([], [], color='k', lw=LW, linestyle=LSTYLES[s],
           label=STATE_LABELS[s])
    for s in ["Z", "X"]
]

leg1 = ax.legend(handles=dist_handles, title='Distance',
                 title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
                 loc='lower right', frameon=True, framealpha=0.5)
ax.add_artist(leg1)
ax.legend(handles=state_handles,
          title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
          loc='upper left', frameon=True, framealpha=0.5)

ax.set_title(f'Bell ZZ-LS Teleportation — LER vs Routing Distance ($p = 10^{{-3}}$)',
             fontweight='bold', fontsize=FS_TITLE)

fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
