"""
Two-qubit ZZ LS teleportation — LER vs p, Z vs X channel comparison.

Two panels: teleport_Z (left) and teleport_X (right).
Each panel: d=3,5,7 lines colored by distance.
A third inset-style panel shows the Z/X ratio vs p.

Output: results/zz_tele.png

Usage:
    venv/bin/python eval/logical_circuit_benchmark/two-qubit-teleportation/plot_zz_tele.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

CSV = Path("eval/logical_circuit_benchmark/two-qubit-teleportation/results/zz_tele_results.csv")
OUT = Path("eval/logical_circuit_benchmark/two-qubit-teleportation/results/zz_tele.png")

DISTS   = [3, 5, 7]
MARKERS = {3: "o", 5: "s", 7: "^"}
LW, MS  = 1.8, 6

FS_TITLE  = 10
FS_LABEL  = 10
FS_TICK   = 9
FS_LEGEND = 9

df = pd.read_csv(CSV)

fig, (ax_ler, ax_ratio) = plt.subplots(1, 2, figsize=(6.0, 3.4), constrained_layout=True)

STATE_LS  = {'teleport_Z': '-', 'teleport_X': '--'}
STATE_LBL = {'teleport_Z': '$|Z\\rangle$', 'teleport_X': '$|X\\rangle$'}

dist_handles = []
for d in DISTS:
    color  = PALETTE_DISTANCE[d]
    marker = MARKERS[d]

    sub_z = df[(df.d == d) & (df.state == 'teleport_Z')].sort_values('p')
    sub_x = df[(df.d == d) & (df.state == 'teleport_X')].sort_values('p')

    for sub, ls in [(sub_z, '-'), (sub_x, '--')]:
        ax_ler.loglog(sub['p'], sub['logical_error_rate'],
                      marker=marker, color=color, lw=LW, ms=MS,
                      markeredgecolor='none', linestyle=ls)

    # Ratio X/Z
    p_common = np.intersect1d(sub_z['p'].values, sub_x['p'].values)
    r_z = sub_z[sub_z['p'].isin(p_common)].sort_values('p')['logical_error_rate'].values
    r_x = sub_x[sub_x['p'].isin(p_common)].sort_values('p')['logical_error_rate'].values
    ax_ratio.semilogx(p_common, r_x / r_z,
                      marker=marker, color=color, lw=LW, ms=MS,
                      markeredgecolor='none')

    from matplotlib.lines import Line2D
    dist_handles.append(Line2D([], [], color=color, marker=marker, lw=LW, ms=MS,
                               markeredgecolor='none', label=f'$d={d}$'))

# LER panel
ax_ler.set_xlabel('$p$', fontsize=FS_LABEL)
ax_ler.set_ylabel('LER', fontsize=FS_LABEL)
ax_ler.set_title('Teleport $|Z\\rangle$ vs $|X\\rangle$', fontsize=FS_TITLE)
ax_ler.tick_params(labelsize=FS_TICK)
ax_ler.set_xlim(3e-4, 2e-2)
ax_ler.grid(True, which='major', ls='--', alpha=0.5)
bold_ticks(ax_ler)

from matplotlib.lines import Line2D
state_handles = [Line2D([], [], color='k', lw=LW, linestyle=ls, label=lbl)
                 for ls, lbl in [('-', '$|Z\\rangle$'), ('--', '$|X\\rangle$')]]
leg1 = ax_ler.legend(handles=dist_handles, title='Distance',
                     title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
                     loc='upper left', frameon=True, framealpha=0.4)
ax_ler.add_artist(leg1)
ax_ler.legend(handles=state_handles, title='State',
              title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
              loc='lower right', frameon=True, framealpha=0.4)

# Ratio panel
ax_ratio.axhline(1.0, color='gray', lw=1.0, ls='--', alpha=0.6)
ax_ratio.set_xlabel('$p$', fontsize=FS_LABEL)
ax_ratio.set_title('LER$_X$ / LER$_Z$', fontsize=FS_TITLE)
ax_ratio.set_ylabel('Ratio', fontsize=FS_LABEL)
ax_ratio.tick_params(labelsize=FS_TICK)
ax_ratio.grid(True, which='major', ls='--', alpha=0.5)
ax_ratio.set_xlim(3e-4, 2e-2)
ax_ratio.set_ylim(0, 3.0)
bold_ticks(ax_ratio)
ax_ratio.legend(handles=dist_handles, title='Distance',
                title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
                loc='upper right', frameon=True, framealpha=0.4)

fig.suptitle('Two-Qubit ZZ LS Teleportation', fontweight='bold', fontsize=FS_TITLE + 1)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
