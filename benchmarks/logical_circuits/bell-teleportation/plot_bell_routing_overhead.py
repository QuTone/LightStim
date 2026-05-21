"""
Bell ZZ-LS teleportation — Routing distance overhead for Teleport |Z⟩ state.

X-axis: routing distance (1×, 2×, 4×, 8×)
Y-axis: LER(routing_mult) / LER(1×)   [normalized to 1× baseline]
Lines:  d=3,5,7  ×  p=5e-4, 1e-3, 5e-3  → 9 lines total

Output: results/bell_routing_overhead.png

Usage:
    venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_bell_routing_overhead.py
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

from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

DIST_CSV = Path("benchmarks/logical_circuits/bell-teleportation/results/ls_zz_dist_results.csv")
BASE_CSV = Path("benchmarks/logical_circuits/bell-teleportation/results/ls_zz_results.csv")
OUT      = Path("benchmarks/logical_circuits/bell-teleportation/results/bell_routing_overhead.png")

DISTS   = [3, 5, 7]
MULTS   = [1, 2, 4, 8]
P_VALS  = [5e-4, 1e-3]
STATE   = 'Z'

MARKERS  = {3: "o",  5: "s",  7: "^"}
LSTYLES  = {5e-4: "-", 1e-3: "--", 5e-3: "-."}
P_LABELS = {5e-4: "$5\\times10^{-4}$", 1e-3: "$10^{-3}$", 5e-3: "$5\\times10^{-3}$"}

LW, MS = 2.0, 7
FS_TITLE  = 11
FS_LABEL  = 11
FS_TICK   = 10
FS_LEGEND = 9

# ── Load & combine ────────────────────────────────────────────────────────────
df_dist = pd.read_csv(DIST_CSV)
df_base = pd.read_csv(BASE_CSV)
df_base['routing_mult'] = 1

df_all = pd.concat([
    df_base[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
    df_dist[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
], ignore_index=True)
df_z = df_all[df_all['state'] == STATE]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)

for d in DISTS:
    color = PALETTE_DISTANCE[d]
    marker = MARKERS[d]
    for p in P_VALS:
        ls = LSTYLES[p]
        # baseline at 1×
        base = df_z[(df_z.d == d) & (df_z.routing_mult == 1) & (df_z.p == p)]
        if base.empty:
            continue
        base_ler = base['logical_error_rate'].values[0]

        overheads = []
        for mult in MULTS:
            row = df_z[(df_z.d == d) & (df_z.routing_mult == mult) & (df_z.p == p)]
            overheads.append(row['logical_error_rate'].values[0] / base_ler
                             if not row.empty else np.nan)

        ax.plot(MULTS, overheads,
                color=color, marker=marker, lw=LW, ms=MS,
                markeredgecolor='none', linestyle=ls)

ax.axhline(1.0, color='gray', lw=1.0, ls='--', alpha=0.5)
ax.set_xticks(MULTS)
ax.set_xticklabels([f'{m}×' for m in MULTS])
ax.set_xlabel('Routing distance', fontsize=FS_LABEL)
ax.set_ylabel('LER / LER$_{1\\times}$', fontsize=FS_LABEL)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, which='major', ls='--', alpha=0.5)
bold_ticks(ax)

# Two legend groups
dist_handles = [Line2D([], [], color=PALETTE_DISTANCE[d], marker=MARKERS[d],
                       lw=LW, ms=MS, markeredgecolor='none', label=f'$d={d}$')
                for d in DISTS]
p_handles = [Line2D([], [], color='k', lw=LW, linestyle=LSTYLES[p],
                    label=P_LABELS[p])
             for p in P_VALS]

leg1 = ax.legend(handles=dist_handles, title='Distance',
                 title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
                 loc='upper left', frameon=True, framealpha=0.4)
ax.add_artist(leg1)
ax.legend(handles=p_handles, title='$p$',
          title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
          loc='center left', bbox_to_anchor=(0.01, 0.55),
          frameon=True, framealpha=0.4)

ax.set_title('Teleport $|Z\\rangle$ — Routing Overhead', fontweight='bold',
             fontsize=FS_TITLE)

fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
