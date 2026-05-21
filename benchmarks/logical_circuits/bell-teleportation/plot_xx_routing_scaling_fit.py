"""
XX-LS routing scaling: LER vs routing multiplier for d=7, p=1e-3.
Symmetric counterpart to plot_routing_scaling_fit.py (ZZ-LS).

Expected behavior (X↔Z dual of ZZ-LS):
  Teleport |X⟩ — LER grows ~linearly with r  (X errors silent in XX coupler)
  Teleport |Z⟩ — LER grows sub-linearly with r (Z errors detected by XX coupler)

Usage:
    venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_xx_routing_scaling_fit.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

from lightstim.plot.styles import apply_paper_style, bold_ticks

apply_paper_style()

DIST_CSV = Path("benchmarks/logical_circuits/bell-teleportation/results/ls_xx_dist_results.csv")
BASE_CSV = Path("benchmarks/logical_circuits/bell-teleportation/results/ls_xx_results.csv")
OUT      = Path("benchmarks/logical_circuits/bell-teleportation/results/xx_routing_scaling_fit.png")

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
# Deduplicate: keep one row per (state, routing_mult)
df7 = df7.drop_duplicates(subset=['state', 'routing_mult'])

def get_series(state):
    sub = df7[df7['state'] == state].sort_values('routing_mult')
    return sub['routing_mult'].values.astype(float), sub['logical_error_rate'].values

r_X, ler_X = get_series('X')
r_Z, ler_Z = get_series('Z')

print(f"X  r={list(r_X.astype(int))}  LER={[f'{v:.2e}' for v in ler_X]}")
print(f"Z  r={list(r_Z.astype(int))}  LER={[f'{v:.2e}' for v in ler_Z]}")

# ── Linear fit for |X⟩ (expected linear) ────────────────────────────────────
if len(r_X) >= 2:
    slope_X, intercept_X, r2_X, _, _ = linregress(r_X, ler_X)
    r2_X = r2_X ** 2  # linregress returns r, we want R²
    print(f"\n|X⟩ linear fit: LER = {slope_X:.3e}·r + {intercept_X:.3e}  (R²={r2_X:.3f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)

COLOR_X = '#d62728'
COLOR_Z = '#1f77b4'

ax.plot(r_X, ler_X * 1e3, 's-', color=COLOR_X, lw=2, ms=7,
        markeredgecolor='k', markeredgewidth=0.4,
        label=r'Teleport $|X\rangle$')

ax.plot(r_Z, ler_Z * 1e3, 'o--', color=COLOR_Z, lw=2, ms=7,
        markeredgecolor='k', markeredgewidth=0.4,
        label=r'Teleport $|Z\rangle$')

# Linear guide for |X⟩
if len(r_X) >= 2 and slope_X > 0:
    r_fit = np.linspace(r_X.min(), r_X.max(), 100)
    ler_fit = slope_X * r_fit + intercept_X
    ax.plot(r_fit, ler_fit * 1e3, '--', color=COLOR_X, lw=1.0, alpha=0.4,
            label=f'Linear fit (R²={r2_X:.2f})')

ax.set_xlabel('Routing multiplier $r$', fontsize=11)
ax.set_ylabel(r'LER  ($\times 10^{-3}$)', fontsize=11)
ax.set_title(f'Bell XX-LS  —  LER vs Routing Distance\n($d=7$, $p=10^{{-3}}$)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10, frameon=True, framealpha=0.6)
ax.grid(True, ls='--', alpha=0.4)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
bold_ticks(ax)

fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
