"""
Plot full circuit-level noise sweep for TG and LS 7-to-1 distillation.

Reads:
    tg_7to1/TG_full_noise_results.csv
    ls_7to1/LS_full_noise_results.csv

Outputs (saved to results/):
    fig_full_noise_ls.png         — LS only, P_out vs p, d=3,5,7
    fig_full_noise_tg.png         — TG only, P_out vs p, d=3,5,7
    fig_full_noise_both.png       — TG + LS side-by-side

Upper-bound rows (errors == 0 or shots >= max_shots with errors < 5)
are plotted as downward triangles with dashed stems.

Usage (from repo root):
    venv/bin/python eval/logical_circuit_benchmark/distillation/plot_full_noise.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))

from lightstim.plot.styles import (
    apply_paper_style, bold_ticks, PALETTE_DISTANCE,
)

apply_paper_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
TG_CSV  = os.path.join(HERE, 'tg_7to1', 'TG_full_noise_results.csv')
LS_CSV  = os.path.join(HERE, 'ls_7to1', 'LS_full_noise_results.csv')
OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        print(f'  [skip] {path}  — not found')
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f'  Loaded {len(df)} rows ← {os.path.basename(path)}')
    return df

print('Loading...')
tg = load(TG_CSV)
ls = load(LS_CSV)

# ── Plot helpers ──────────────────────────────────────────────────────────────
MARKERS    = ['o', 's', '^', 'D', 'v']
UB_MARKER  = 'v'          # downward triangle for upper bounds
MAX_SHOTS  = 100_000_000
UB_THRESH  = 5            # treat rows with errors < this as upper bounds

def wilson_errbar(df):
    """3-sigma Wilson CI on ler_ps. Returns (lo, hi) aligned with df.index."""
    n = (df['shots'] * df['post_selection_rate']).clip(lower=1)
    k = df['errors'].astype(float)
    p_hat = k / n
    z = 3.0
    denom  = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    lo = (centre - half).clip(lower=0)
    hi = centre + half
    return (p_hat - lo).values, (hi - p_hat).values


def plot_full_noise(ax, df, title='', add_legend=True):
    """Plot P_out vs p for each distance d in df."""
    if df.empty:
        ax.set_visible(False)
        return

    distances = sorted(df['d'].unique())
    for i, d in enumerate(distances):
        sub = df[df['d'] == d].sort_values('p').copy()
        color  = PALETTE_DISTANCE.get(d, PALETTE_DISTANCE.get(3))
        marker = MARKERS[i % len(MARKERS)]
        label  = f'$d={d}$'

        # True upper bound: errors == 0 (LER could only be lower)
        is_ub = sub['errors'] == 0
        meas  = sub[~is_ub]
        ub    = sub[is_ub]

        # Measured points with error bars (includes shots=max_shots with errors>0)
        if not meas.empty:
            lo, hi = wilson_errbar(meas)
            ax.errorbar(meas['p'], meas['ler_ps'],
                        yerr=[lo, hi],
                        fmt=marker + '-', color=color, capsize=3,
                        lw=2, ms=7, markeredgecolor='k', markeredgewidth=0.4,
                        label=label)

        # Zero-error upper bounds: plot at 1/shots with downward arrow
        if not ub.empty:
            ub_val = 3.0 / ub['shots']   # 3-sigma upper bound: ~3/N
            first_ub = True
            for (_, row), uv in zip(ub.iterrows(), ub_val):
                lbl = label if (meas.empty and first_ub) else '_nolegend_'
                ax.annotate('',
                            xy=(row['p'], uv * 0.25),
                            xytext=(row['p'], uv),
                            arrowprops=dict(arrowstyle='-|>',
                                            color=color, lw=1.8,
                                            mutation_scale=10))
                ax.plot(row['p'], uv, UB_MARKER, color=color, ms=7,
                        markeredgecolor='k', markeredgewidth=0.4,
                        alpha=0.65, label=lbl)
                first_ub = False

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Physical error rate $p$', fontsize=14)
    ax.set_ylabel(r'$P_{\mathrm{out}}$', fontsize=14)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if add_legend:
        ax.legend(fontsize=11, framealpha=0.85, loc='upper left')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    bold_ticks(ax)


# ── Figure 1: LS only ─────────────────────────────────────────────────────────
if not ls.empty:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    plot_full_noise(ax, ls, title='LS 7-to-1 — Full circuit-level noise')
    out = os.path.join(OUT_DIR, 'fig_full_noise_ls.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

# ── Figure 2: TG only ─────────────────────────────────────────────────────────
if not tg.empty:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    plot_full_noise(ax, tg, title='TG 7-to-1 — Full circuit-level noise')
    out = os.path.join(OUT_DIR, 'fig_full_noise_tg.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

# ── Figure 3: TG + LS side-by-side ────────────────────────────────────────────
if not tg.empty or not ls.empty:
    n_panels = int(not tg.empty) + int(not ls.empty)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                             constrained_layout=True, sharey=False)
    if n_panels == 1:
        axes = [axes]

    panel_data = []
    if not tg.empty:
        panel_data.append((tg, 'TG 7-to-1'))
    if not ls.empty:
        panel_data.append((ls, 'LS 7-to-1'))

    for ax, (df, title) in zip(axes, panel_data):
        plot_full_noise(ax, df, title=title, add_legend=True)

    # Shared legend from first panel only (distances are the same)
    out = os.path.join(OUT_DIR, 'fig_full_noise_both.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

print('\nDone.')
