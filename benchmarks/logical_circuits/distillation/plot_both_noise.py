"""
Plot 'both' noise mode: injection + fixed circuit-level noise (p=5e-4).

Shows P_out vs P_in on the same axes as injection-only, revealing how
Clifford circuit noise raises the floor above the 7*P_in^3 theory line.

Reads:
    tg_7to1/TG_injection_results.csv    — injection-only baseline (on 7p^3)
    ls_7to1/LS_injection_results.csv
    tg_7to1/TG_both_results.csv         — injection + p_circuit=5e-4
    ls_7to1/LS_both_results.csv

Output:
    results/fig_both_noise_tg.png
    results/fig_both_noise_ls.png
    results/fig_both_noise_combined.png

Usage (from repo root):
    venv/bin/python benchmarks/logical_circuits/distillation/plot_both_noise.py
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))

from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE_DISTANCE

apply_paper_style()

OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

P_CIRCUIT = 5e-4   # fixed circuit-level noise used in 'both' runs

# ── Load ──────────────────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df['p_in'] > 0].copy()
    return df

tg_inj  = load(os.path.join(HERE, 'tg_7to1', 'TG_injection_results.csv'))
ls_inj  = load(os.path.join(HERE, 'ls_7to1', 'LS_injection_results.csv'))
tg_both = load(os.path.join(HERE, 'tg_7to1', 'TG_both_results.csv'))
ls_both = load(os.path.join(HERE, 'ls_7to1', 'LS_both_results.csv'))

# Filter 'both' to only p = P_CIRCUIT rows
if not tg_both.empty:
    tg_both = tg_both[np.isclose(tg_both['p'], P_CIRCUIT)].copy()
if not ls_both.empty:
    ls_both = ls_both[np.isclose(ls_both['p'], P_CIRCUIT)].copy()

MARKERS   = ['o', 's', '^', 'D', 'v']
UB_THRESH = 5

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


def plot_panel(ax, df_inj, df_both, title=''):
    """Plot injection-only + both-noise data on one axis."""
    if df_inj.empty and df_both.empty:
        ax.set_visible(False)
        return

    all_pin, all_ler = [], []
    distances = sorted(set(
        list(df_inj['d'].unique() if not df_inj.empty else []) +
        list(df_both['d'].unique() if not df_both.empty else [])
    ))

    for i, d in enumerate(distances):
        color  = PALETTE_DISTANCE.get(d, PALETTE_DISTANCE.get(3))
        marker = MARKERS[i % len(MARKERS)]

        # ── Injection-only (open markers, dashed line) ──
        if not df_inj.empty:
            sub = df_inj[df_inj['d'] == d].sort_values('p_in')
            meas = sub[sub['errors'] >= UB_THRESH]
            if not meas.empty:
                lo, hi = wilson_errbar(meas)
                ax.errorbar(meas['p_in'], meas['ler_ps'],
                            yerr=[lo, hi],
                            fmt=marker + '--', color=color, capsize=2,
                            lw=1.4, ms=7,
                            markeredgecolor='k', markeredgewidth=0.4,
                            markerfacecolor='white',
                            label=f'$d={d}$ (inj only)',
                            zorder=3)
                all_pin.extend(meas['p_in'].tolist())
                all_ler.extend(meas['ler_ps'].tolist())

        # ── Both noise (filled markers, solid line) ──
        if not df_both.empty:
            sub2 = df_both[df_both['d'] == d].sort_values('p_in')
            meas2 = sub2[sub2['errors'] >= UB_THRESH]
            ub2   = sub2[sub2['errors'] < UB_THRESH]
            if not meas2.empty:
                lo2, hi2 = wilson_errbar(meas2)
                ax.errorbar(meas2['p_in'], meas2['ler_ps'],
                            yerr=[lo2, hi2],
                            fmt=marker + '-', color=color, capsize=2,
                            lw=1.8, ms=8,
                            markeredgecolor='k', markeredgewidth=0.5,
                            label=f'$d={d}$ ($p_{{\\rm circ}}={P_CIRCUIT:.0e}$)',
                            zorder=4)
                all_pin.extend(meas2['p_in'].tolist())
                all_ler.extend(meas2['ler_ps'].tolist())
            for (_, row) in ub2.iterrows():
                uv = 3.0 / max(row['shots'] * row['post_selection_rate'], 1)
                ax.annotate('', xy=(row['p_in'], uv*0.25), xytext=(row['p_in'], uv),
                            arrowprops=dict(arrowstyle='-|>', color=color,
                                           lw=1.5, mutation_scale=9))
                ax.plot(row['p_in'], uv, 'v', color=color, ms=7, alpha=0.7,
                        markeredgecolor='k', markeredgewidth=0.4)

    # Theory line
    if all_pin:
        xlo = 10 ** (np.log10(min(all_pin)) - 0.5)
        xhi = 10 ** (np.log10(max(all_pin)) + 0.3)
        ylo = 10 ** (np.log10(min(all_ler)) - 1.2)
        yhi = min(1.0, 10 ** (np.log10(max(all_ler)) + 0.5))
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        pin_th = np.logspace(np.log10(max(xlo, 1e-12)), np.log10(xhi), 300)
        ax.plot(pin_th, 7 * pin_th**3, 'k--', lw=1.8, label=r'$7\,P_{\rm in}^3$', zorder=0)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$P_{\rm in}$', fontsize=13)
    ax.set_ylabel(r'$P_{\rm out}$', fontsize=13)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.85, loc='upper left', ncol=1)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    bold_ticks(ax)


# ── Figure 1: TG ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
plot_panel(ax, tg_inj, tg_both, title=f'TG 7-to-1  ($p_{{\\rm circ}}={P_CIRCUIT:.0e}$)')
fig.savefig(os.path.join(OUT_DIR, 'fig_both_noise_tg.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: fig_both_noise_tg.png')

# ── Figure 2: LS ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
plot_panel(ax, ls_inj, ls_both, title=f'LS 7-to-1  ($p_{{\\rm circ}}={P_CIRCUIT:.0e}$)')
fig.savefig(os.path.join(OUT_DIR, 'fig_both_noise_ls.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: fig_both_noise_ls.png')

# ── Figure 3: side-by-side ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
plot_panel(axes[0], tg_inj, tg_both, title='TG 7-to-1')
plot_panel(axes[1], ls_inj, ls_both, title='LS 7-to-1')
out = os.path.join(OUT_DIR, 'fig_both_noise_combined.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

print('Done.')
