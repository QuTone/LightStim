"""
Plot injection-only noise for TG and LS 7-to-1 distillation.

Reads:
    tg_7to1/TG_injection_results.csv
    ls_7to1/LS_injection_results.csv

Outputs (saved to results/):
    fig_injection_ls.png          — LS only, P_out vs P_in with 7*P_in^3 theory
    fig_injection_tg.png          — TG only
    fig_injection_both.png        — TG + LS side-by-side (publication figure)

Usage (from repo root):
    venv/bin/python eval/logical_circuit_benchmark/distillation/plot_injection.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))

from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE_DISTANCE

apply_paper_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
TG_CSV  = os.path.join(HERE, 'tg_7to1', 'TG_injection_results.csv')
LS_CSV  = os.path.join(HERE, 'ls_7to1', 'LS_injection_results.csv')
OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        print(f'  [skip] {os.path.basename(path)}  — not found')
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Only rows with a valid p_in measurement
    df = df[df['p_in'] > 0].copy()
    print(f'  Loaded {len(df)} valid rows ← {os.path.basename(path)}')
    return df

print('Loading...')
tg = load(TG_CSV)
ls = load(LS_CSV)

# ── Helpers ───────────────────────────────────────────────────────────────────
MARKERS   = ['o', 's', '^', 'D', 'v']
UB_THRESH = 5      # rows with fewer errors than this are upper bounds

def wilson_errbar(df):
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


def add_theory(ax, label=r'$7\,P_{\rm in}^3$'):
    """Draw 7*P_in^3 spanning the plot's current x limits."""
    xlo, xhi = ax.get_xlim()
    pin = np.logspace(np.log10(max(xlo, 1e-10)), np.log10(xhi), 300)
    ax.plot(pin, 7 * pin**3, 'k--', lw=1.8, label=label, zorder=0)


def plot_injection(ax, df, title='', add_legend=True):
    if df.empty:
        ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                transform=ax.transAxes, fontsize=13, color='gray')
        ax.set_title(title, fontsize=14, fontweight='bold')
        return

    distances = sorted(df['d'].unique())
    all_pin_meas, all_ler_meas = [], []

    for i, d in enumerate(distances):
        sub   = df[df['d'] == d].sort_values('p_in')
        color  = PALETTE_DISTANCE.get(d, PALETTE_DISTANCE.get(3))
        marker = MARKERS[i % len(MARKERS)]
        label  = f'$d={d}$'

        is_ub = sub['errors'] < UB_THRESH
        meas  = sub[~is_ub]
        ub    = sub[is_ub]

        # Measured points with 3-sigma error bars
        if not meas.empty:
            lo, hi = wilson_errbar(meas)
            ax.errorbar(meas['p_in'], meas['ler_ps'],
                        yerr=[lo, hi],
                        fmt=marker + '-', color=color, capsize=3,
                        lw=2, ms=7, markeredgecolor='k', markeredgewidth=0.4,
                        label=label)
            all_pin_meas.extend(meas['p_in'].tolist())
            all_ler_meas.extend(meas['ler_ps'].tolist())

        # Upper bounds: downward arrow at 3/N_post
        if not ub.empty:
            n_post = (ub['shots'] * ub['post_selection_rate']).clip(lower=1)
            ub_val = 3.0 / n_post  # 3-sigma upper bound
            first  = True
            for (_, row), uv in zip(ub.iterrows(), ub_val):
                lbl = label if (meas.empty and first) else '_nolegend_'
                ax.annotate('',
                            xy=(row['p_in'], uv * 0.20),
                            xytext=(row['p_in'], uv),
                            arrowprops=dict(arrowstyle='-|>',
                                            color=color, lw=1.8,
                                            mutation_scale=10))
                ax.plot(row['p_in'], uv, 'v', color=color, ms=7, alpha=0.7,
                        markeredgecolor='k', markeredgewidth=0.4, label=lbl)
                first = False

    # Set sensible axis limits from measured data
    if all_pin_meas and all_ler_meas:
        xlo = 10 ** (np.log10(min(all_pin_meas)) - 0.5)
        xhi = 10 ** (np.log10(max(all_pin_meas)) + 0.3)
        ylo = 10 ** (np.log10(min(all_ler_meas)) - 1.0)
        yhi = min(1.0, 10 ** (np.log10(max(all_ler_meas)) + 0.5))
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    add_theory(ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$P_{\rm in}$', fontsize=14)
    ax.set_ylabel(r'$P_{\rm out}$', fontsize=14)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if add_legend:
        ax.legend(fontsize=11, framealpha=0.85, loc='upper left')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    bold_ticks(ax)


# ── Figure 1: LS only ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
plot_injection(ax, ls, title='LS 7-to-1 — Injection-only noise')
fig.savefig(os.path.join(OUT_DIR, 'fig_injection_ls.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_DIR}/fig_injection_ls.png')

# ── Figure 2: TG only ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
plot_injection(ax, tg, title='TG 7-to-1 — Injection-only noise')
fig.savefig(os.path.join(OUT_DIR, 'fig_injection_tg.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_DIR}/fig_injection_tg.png')

# ── Figure 3: TG + LS side-by-side (primary publication figure) ───────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

plot_injection(axes[0], tg, title='TG 7-to-1', add_legend=True)
plot_injection(axes[1], ls, title='LS 7-to-1', add_legend=True)

# Shared y-axis range if both have data
if not tg.empty and not ls.empty:
    all_y = ([y for line in axes[0].get_lines() for y in line.get_ydata()
               if np.isfinite(y) and y > 0] +
             [y for line in axes[1].get_lines() for y in line.get_ydata()
               if np.isfinite(y) and y > 0])
    if all_y:
        ymin = 10 ** (np.log10(min(all_y)) - 0.5)
        ymax = 10 ** (np.log10(max(all_y)) + 0.5)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

out = os.path.join(OUT_DIR, 'fig_injection_both.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# ── Figure 4: TG + LS merged into one compact panel ───────────────────────────
fig, ax = plt.subplots(figsize=(3.0, 4.2), constrained_layout=True)

all_pin, all_ler = [], []

for df, marker in [(tg, 'o'), (ls, 's')]:
    if df.empty:
        continue
    for i, d in enumerate(sorted(df['d'].unique())):
        sub   = df[df['d'] == d].sort_values('p_in')
        color = PALETTE_DISTANCE.get(d, PALETTE_DISTANCE.get(3))
        meas  = sub[sub['errors'] >= UB_THRESH]
        ub    = sub[sub['errors'] < UB_THRESH]

        if not meas.empty:
            lo, hi = wilson_errbar(meas)
            ax.errorbar(meas['p_in'], meas['ler_ps'],
                        yerr=[lo, hi],
                        fmt=marker, color=color, capsize=2,
                        linestyle='none', lw=1.5, ms=9,
                        markeredgecolor='k', markeredgewidth=0.4,
                        label='_nolegend_')
            all_pin.extend(meas['p_in'].tolist())
            all_ler.extend(meas['ler_ps'].tolist())


# Add proxy legend entries: d colours + protocol markers
import matplotlib.lines as mlines
handles = []
for d in sorted(tg['d'].unique() if not tg.empty else ls['d'].unique()):
    color = PALETTE_DISTANCE.get(d, PALETTE_DISTANCE.get(3))
    handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                 ms=9, markeredgecolor='k', markeredgewidth=0.4,
                                 label=f'$d={d}$'))
handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                              ms=9, markeredgecolor='k', markeredgewidth=0.4,
                              label='TG'))
handles.append(mlines.Line2D([], [], color='gray', marker='s', linestyle='None',
                              ms=9, markeredgecolor='k', markeredgewidth=0.4,
                              label='LS'))

# Theory line
if all_pin:
    xlo = 10 ** (np.log10(min(all_pin)) - 0.4)
    xhi = 10 ** (np.log10(max(all_pin)) + 0.3)
    ylo = 10 ** (np.log10(min(all_ler)) - 1.0)
    yhi = min(1.0, 10 ** (np.log10(max(all_ler)) + 0.5))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    pin_th = np.logspace(np.log10(max(xlo, 1e-10)), np.log10(xhi), 300)
    theory_line, = ax.plot(pin_th, 7 * pin_th**3, 'k--', lw=1.8, zorder=0)
    handles = [mlines.Line2D([], [], color='k', linestyle='--', lw=1.8,
                              label=r'$7\,P_{\rm in}^3$')] + handles

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$P_{\rm in}$', fontsize=12)
ax.set_ylabel(r'$P_{\rm out}$', fontsize=12)
ax.set_title('Distillation LER vs $p_{\\rm inject}$', fontsize=10, fontweight='bold')
ax.legend(handles=handles, fontsize=9, framealpha=0.85,
          loc='upper left', handlelength=1.2, handletextpad=0.4,
          borderpad=0.5, labelspacing=0.3)
ax.grid(True, which='both', ls='--', alpha=0.4)
ax.tick_params(axis='both', labelsize=8)
bold_ticks(ax)

out = os.path.join(OUT_DIR, 'fig_injection_combined.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

print('\nDone.')
