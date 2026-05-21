"""
Fit and plot LER = c_d · r · p^{(d+1)/2} for vulnerable states.

Left panel:  ZZ-LS Teleport |Z⟩
Right panel: XX-LS Teleport |X⟩

Each panel: LER/r  vs  p (log-log), grouped by d.
Theory slope (d+1)/2 shown as dashed guide.

Usage:
    venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_ler_fit.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

P_MAX  = 2e-3   # only fit / plot below this (sub-threshold regime)
OUT    = Path("benchmarks/logical_circuits/bell-teleportation/results/ler_fit.png")

DISTANCES = [3, 5, 7]
MARKERS   = {3: 'o', 5: 's', 7: '^'}


def load(dist_csv, base_csv, state):
    dist = pd.read_csv(dist_csv)
    base = pd.read_csv(base_csv)
    base['routing_mult'] = 1
    df = pd.concat([
        base[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
        dist[['d', 'state', 'routing_mult', 'p', 'logical_error_rate']],
    ], ignore_index=True)
    df = df[(df['state'] == state) &
            (df['logical_error_rate'] > 0) &
            (df['logical_error_rate'] < 0.4)]
    return df.drop_duplicates(subset=['d', 'routing_mult', 'p'])


zz = load("benchmarks/logical_circuits/bell-teleportation/results/ls_zz_dist_results.csv",
          "benchmarks/logical_circuits/bell-teleportation/results/ls_zz_results.csv", "Z")
xx = load("benchmarks/logical_circuits/bell-teleportation/results/ls_xx_dist_results.csv",
          "benchmarks/logical_circuits/bell-teleportation/results/ls_xx_results.csv", "X")

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5),
                         sharey=False, constrained_layout=True)

panels = [
    (axes[0], zz, 'ZZ-LS  (Teleport $|Z\\rangle$)'),
    (axes[1], xx, 'XX-LS  (Teleport $|X\\rangle$)'),
]

fit_results = {}
for ax, df, title in panels:
    sub = df[df['p'] <= P_MAX].copy()
    key = title[:2]
    fit_results[key] = {}

    for d in DISTANCES:
        color = PALETTE_DISTANCE[d]
        marker = MARKERS[d]
        pts = sub[sub['d'] == d].copy()
        if pts.empty:
            continue

        # Y-axis: LER / r  (removes the linear-r dependence)
        y = pts['logical_error_rate'] / pts['routing_mult']
        x = pts['p']

        # Scatter: one point per (p, r) combo → group by p and plot median
        by_p = pts.groupby('p').apply(
            lambda g: (g['logical_error_rate'] / g['routing_mult']).median()
        ).reset_index()
        by_p.columns = ['p', 'ler_over_r']

        ax.loglog(by_p['p'], by_p['ler_over_r'],
                  marker=marker, color=color, ms=6, lw=0,
                  markeredgecolor='k', markeredgewidth=0.4,
                  label=f'$d={d}$')

        # Fit log(LER/r) = log(c) + k*log(p)
        log_x = np.log(by_p['p'])
        log_y = np.log(by_p['ler_over_r'])
        slope, intercept, r2, _, _ = linregress(log_x, log_y)
        c_fit = np.exp(intercept)
        fit_results[key][d] = {'c': c_fit, 'k': slope, 'R2': r2**2}

        # Fitted line
        p_line = np.logspace(np.log10(by_p['p'].min()),
                             np.log10(by_p['p'].max()), 80)
        ax.loglog(p_line, c_fit * p_line**slope, '-', color=color, lw=1.5, alpha=0.8)

        print(f"  {key} d={d}: k={slope:.3f} (theory {(d+1)/2:.1f}), "
              f"c={c_fit:.2e}, R²={r2**2:.3f}")

    # Theory guide lines (dashed, no label)
    p_ref = np.array([P_MAX * 0.1, P_MAX])
    for d in DISTANCES:
        color = PALETTE_DISTANCE[d]
        k_theory = (d + 1) / 2
        if d in fit_results[key]:
            c_ref = fit_results[key][d]['c']
            ax.loglog(p_ref, c_ref * p_ref**k_theory, 'k:', lw=0.8, alpha=0.3)

    ax.set_xlabel('Physical error rate $p$', fontsize=9)
    ax.set_ylabel(r'LER$/r$', fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, frameon=True, framealpha=0.7)
    ax.grid(True, which='both', ls='--', alpha=0.35)
    bold_ticks(ax)

# Annotate fitted slopes on the figure
for ax, df, title in panels:
    key = title[:2]
    lines = [f"$d={d}$: $k={fit_results[key][d]['k']:.2f}$  (th. {(d+1)/2:.1f})"
             for d in DISTANCES if d in fit_results[key]]
    txt = '\n'.join(lines)
    ax.text(0.97, 0.05, txt, transform=ax.transAxes,
            fontsize=6.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

fig.suptitle(r'LER/$r$  vs  $p$  —  fit to  LER $= c_d \cdot r \cdot p^{(d+1)/2}$',
             fontsize=9, fontweight='bold')

fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {OUT}')

# Print summary table
print('\n=== Fit summary ===')
print(f"{'':5} {'d':3}  {'k_fit':>7}  {'k_theory':>9}  {'c_d':>12}  {'R²':>6}")
for key in ['ZZ', 'XX']:
    for d in DISTANCES:
        if d not in fit_results.get(key, {}):
            continue
        r = fit_results[key][d]
        print(f"{key:5} {d:3}  {r['k']:>7.3f}  {(d+1)/2:>9.1f}  {r['c']:>12.3e}  {r['R2']:>6.3f}")
