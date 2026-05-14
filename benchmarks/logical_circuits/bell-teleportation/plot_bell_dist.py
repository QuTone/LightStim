"""
Bell ZZ teleportation — LER vs p, routing distance + Z/X state comparison.

Panel 1 (Teleport Z): LER vs p, color=distance, linestyle=routing_mult
Panel 2 (Teleport X): same
Panel 3 (Z/X Ratio):  LER_X / LER_Z vs routing_mult at p=1e-3, one curve per d
                      → shows whether routing distance increases Z/X asymmetry

Output: results/bell_dist.png

Usage:
    venv/bin/python eval/logical_circuit_benchmark/bell-teleportation/plot_bell_dist.py
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

DIST_CSV = Path("eval/logical_circuit_benchmark/bell-teleportation/results/ls_zz_dist_results.csv")
BASE_CSV = Path("eval/logical_circuit_benchmark/bell-teleportation/results/ls_zz_results.csv")
OUT      = Path("eval/logical_circuit_benchmark/bell-teleportation/results/bell_dist.png")

DISTS   = [3, 5, 7]
MULTS   = [1, 2, 4, 8]
P_REF   = 1e-3
MARKERS = {3: "o", 5: "s", 7: "^"}
LSTYLES = {1: "-", 2: "--", 4: "-.", 8: ":"}
LW, MS  = 1.8, 6

FS_TITLE  = 10
FS_LABEL  = 10
FS_TICK   = 9
FS_LEGEND = 8

# ── Load & combine ────────────────────────────────────────────────────────────
df_dist = pd.read_csv(DIST_CSV)
df_base = pd.read_csv(BASE_CSV)
df_base['routing_mult'] = 1

df_all = pd.concat([
    df_base[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
    df_dist[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
], ignore_index=True)

# ── Figure: 3 panels ─────────────────────────────────────────────────────────
fig, (ax_z, ax_x, ax_ratio) = plt.subplots(1, 3, figsize=(9.5, 3.4),
                                            constrained_layout=True)

def plot_ler_panel(ax, state_col, title):
    df_s = df_all[df_all['state'] == state_col]
    for d in DISTS:
        color = PALETTE_DISTANCE[d]
        for mult in MULTS:
            sub = df_s[(df_s.d == d) & (df_s.routing_mult == mult)].sort_values('p')
            if sub.empty:
                continue
            ax.loglog(sub['p'], sub['logical_error_rate'],
                      color=color, lw=LW, ms=MS,
                      marker=MARKERS[d], markeredgecolor='none',
                      linestyle=LSTYLES[mult], alpha=0.9)
    ax.set_xlabel('$p$', fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(3e-4, 2e-2)
    ax.grid(True, which='major', ls='--', alpha=0.5)
    bold_ticks(ax)

plot_ler_panel(ax_z, 'Z', 'Teleport $|Z\\rangle$')
plot_ler_panel(ax_x, 'X', 'Teleport $|X\\rangle$')
ax_z.set_ylabel('LER', fontsize=FS_LABEL)

# Shared legends on ax_z
dist_handles = [Line2D([], [], color=PALETTE_DISTANCE[d], marker=MARKERS[d],
                       lw=LW, ms=MS, markeredgecolor='none', label=f'$d={d}$')
                for d in DISTS]
mult_handles = [Line2D([], [], color='k', lw=LW, linestyle=LSTYLES[m], label=f'{m}×')
                for m in MULTS]
leg1 = ax_z.legend(handles=dist_handles, title='Distance',
                   title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
                   loc='upper left', frameon=True, framealpha=0.4)
ax_z.add_artist(leg1)
ax_z.legend(handles=mult_handles, title='Routing',
            title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
            loc='lower right', frameon=True, framealpha=0.4)

# ── Right: X/Z ratio vs routing_mult at P_REF ────────────────────────────────
for d in DISTS:
    color  = PALETTE_DISTANCE[d]
    marker = MARKERS[d]
    ratios = []
    for mult in MULTS:
        z_row = df_all[(df_all.d == d) & (df_all.state == 'Z') &
                       (df_all.routing_mult == mult) & (df_all.p == P_REF)]
        x_row = df_all[(df_all.d == d) & (df_all.state == 'X') &
                       (df_all.routing_mult == mult) & (df_all.p == P_REF)]
        if z_row.empty or x_row.empty:
            ratios.append(np.nan)
        else:
            ratios.append(x_row['logical_error_rate'].values[0] /
                          z_row['logical_error_rate'].values[0])

    ax_ratio.plot(MULTS, ratios,
                  color=color, marker=marker, lw=LW, ms=MS,
                  markeredgecolor='none', label=f'$d={d}$')

ax_ratio.axhline(1.0, color='gray', lw=1.0, ls='--', alpha=0.6)
ax_ratio.set_xlabel('Routing distance (×)', fontsize=FS_LABEL)
ax_ratio.set_ylabel('LER$_X$ / LER$_Z$', fontsize=FS_LABEL)
ax_ratio.set_title(f'$|X\\rangle$/$|Z\\rangle$ Ratio  ($p=10^{{-3}}$)', fontsize=FS_TITLE)
ax_ratio.set_xticks(MULTS)
ax_ratio.set_xticklabels([f'{m}×' for m in MULTS])
ax_ratio.tick_params(labelsize=FS_TICK)
ax_ratio.grid(True, which='major', ls='--', alpha=0.5)
ax_ratio.legend(title='Distance', title_fontsize=FS_LEGEND,
                fontsize=FS_LEGEND, loc='upper left', frameon=True)
bold_ticks(ax_ratio)

fig.suptitle('Bell ZZ-LS Teleportation — Routing Distance & State Comparison',
             fontweight='bold', fontsize=FS_TITLE + 1)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
