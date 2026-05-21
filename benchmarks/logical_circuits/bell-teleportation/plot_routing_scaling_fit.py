"""
Routing scaling: LER vs routing multiplier for d=7, p=1e-3.
Z and X teleportation on same axes, linear LER scale.

Usage:
    venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_routing_scaling_fit.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, bold_ticks

apply_paper_style()

DIST_CSV = Path("benchmarks/logical_circuits/bell-teleportation/results/ls_zz_dist_results.csv")
BASE_CSV = Path("benchmarks/logical_circuits/bell-teleportation/results/ls_zz_results.csv")
OUT      = Path("benchmarks/logical_circuits/bell-teleportation/results/routing_scaling_fit.png")

P_FIX = 1e-3
D     = 7

# ── Load ──────────────────────────────────────────────────────────────────────
df_dist = pd.read_csv(DIST_CSV)
df_base = pd.read_csv(BASE_CSV)
df_base['routing_mult'] = 1

df = pd.concat([
    df_base[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
    df_dist[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
], ignore_index=True)

df7 = df[(df['d'] == D) & np.isclose(df['p'], P_FIX)].copy()

def get_series(state):
    sub = df7[df7['state'] == state].sort_values('routing_mult')
    return sub['routing_mult'].values.astype(float), sub['logical_error_rate'].values

r_Z, ler_Z = get_series('Z')
r_X, ler_X = get_series('X')

print(f"Z  r={list(r_Z.astype(int))}  LER={[f'{v:.2e}' for v in ler_Z]}")
print(f"X  r={list(r_X.astype(int))}  LER={[f'{v:.2e}' for v in ler_X]}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)

COLOR_Z = '#1f77b4'
COLOR_X = '#d62728'

ax.plot(r_Z, ler_Z * 1e3, 'o-', color=COLOR_Z, lw=2, ms=7,
        markeredgecolor='k', markeredgewidth=0.4,
        label=r'Teleport $|Z\rangle$')

ax.plot(r_X, ler_X * 1e3, 's--', color=COLOR_X, lw=2, ms=7,
        markeredgecolor='k', markeredgewidth=0.4,
        label=r'Teleport $|X\rangle$')

ax.set_xlabel('Routing multiplier $r$', fontsize=11)
ax.set_ylabel(r'LER  ($\times 10^{-3}$)', fontsize=11)
ax.set_title(f'Bell ZZ-LS  —  LER vs Routing Distance\n($d=7$, $p=10^{{-3}}$)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10, frameon=True, framealpha=0.6)
ax.grid(True, ls='--', alpha=0.4)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
bold_ticks(ax)

fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
