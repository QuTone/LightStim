"""Plot all logical op benchmark sub-experiment figures.

Format reference: fig1_cnot_ls_subexp.png
All 4 figures unified to the same figsize.

Usage:
    venv/bin/python -m eval.logical_op_benchmark.plot_logical_op
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

RESULTS = Path("eval/logical_op_benchmark/results")
MARKERS  = {3: "o", 5: "s", 7: "^"}

# ── Figure sizes ─────────────────────────────────────────────────────────────
SINGLE_W  = 2.4   # inches — H and S (1 subplot)
MULTI_W   = 8.0   # inches — CNOT LS / trans (5 subplots)
UNIFIED_H = 3.4   # inches — same height for all


def plot_subexps(df, title, out_path, drop_p=None, x_min_override=None,
                 subplot_title_override=None):
    """Canonical sub-experiment plot matching the reference format."""
    drop_p  = drop_p or {}
    subexps = list(df["sub_experiment"].unique())
    ncols   = len(subexps)
    compact = ncols > 1   # 5-panel figures use multi width
    fig_w   = MULTI_W if compact else SINGLE_W

    fig, axes = plt.subplots(1, ncols,
                             figsize=(fig_w, UNIFIED_H),
                             sharey=True,
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    # Compact mode: slightly smaller text for 5 subplots
    fs_title  = 9  if compact else 12
    fs_label  = 9  if compact else 10
    fs_tick   = 8  if compact else 11
    fs_legend = 8  if compact else 10
    lw        = 1.6 if compact else 2.2
    ms        = 6   if compact else 8

    # Compute shared x-limits after drops
    all_p = []
    for subexp in subexps:
        sub  = df[df["sub_experiment"] == subexp]
        excl = drop_p.get(subexp, {})
        for d in sub["d"].unique():
            bad = set(excl.get(d, []))
            all_p.extend(p for p in sub[sub["d"] == d]["p"].unique() if p not in bad)
    x_min = x_min_override if x_min_override is not None else min(all_p) * 0.7
    x_max = max(all_p) * 1.5

    handles, labels = [], []

    for ax, subexp in zip(axes, subexps):
        sub  = df[df["sub_experiment"] == subexp]
        excl = drop_p.get(subexp, {})

        for d in sorted(sub["d"].unique()):
            bad  = set(excl.get(d, []))
            rows = sub[sub["d"] == d].groupby("p")["logical_error_rate"]
            p_vals = np.array(sorted(p for p in rows.groups if p not in bad))
            if len(p_vals) == 0:
                continue
            ler_med = np.array([rows.get_group(p).median() for p in p_vals])
            ler_lo  = np.array([rows.get_group(p).min()    for p in p_vals])
            ler_hi  = np.array([rows.get_group(p).max()    for p in p_vals])

            color = PALETTE_DISTANCE[d]
            line, = ax.loglog(p_vals, ler_med,
                              marker=MARKERS[d], color=color,
                              lw=lw, ms=ms, markeredgecolor="none",
                              label=f"d={d}")
            if (ler_hi - ler_lo).any():
                ax.fill_between(p_vals, ler_lo, ler_hi,
                                color=color, alpha=0.15, linewidth=0)

            if ax is axes[0]:
                handles.append(line)
                labels.append(f"d={d}")

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("$p$", fontsize=fs_label)
        sp_title = (subplot_title_override or {}).get(subexp, subexp)
        ax.set_title(sp_title, fontsize=fs_title)
        ax.tick_params(labelsize=fs_tick)
        bold_ticks(ax)

    axes[0].set_ylabel("LER", fontsize=fs_label)

    if compact:
        axes[-1].legend(handles, labels,
                        title="Distance",
                        title_fontsize=fs_legend,
                        fontsize=fs_legend,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True)
    else:
        axes[-1].legend(handles, labels,
                        title="Distance",
                        title_fontsize=fs_legend,
                        fontsize=fs_legend,
                        loc="lower right",
                        frameon=True)

    fig.suptitle(title, fontweight="bold",
                 fontsize=fs_title + 1)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CNOT LS ZZ-XX protocol ──────────────────────────────────────────────────
df_ls_zz_xx = pd.read_csv(RESULTS / "fig1_cnot_ls_zz_xx_raw.csv")
all_ds = sorted(df_ls_zz_xx["d"].unique())
drop_ls = {s: {d: [1e-4] for d in all_ds} for s in df_ls_zz_xx["sub_experiment"].unique()}
plot_subexps(df_ls_zz_xx,
             "LS CNOT ZZ-XX — Sub-experiments",
             RESULTS / "fig1_cnot_ls_subexp_zz_xx.png",
             drop_p=drop_ls,
             x_min_override=3e-4)

# ── CNOT LS XX-ZZ protocol ──────────────────────────────────────────────────
df_ls_xx_zz = pd.read_csv(RESULTS / "fig1_cnot_ls_xx_zz_raw.csv")
all_ds = sorted(df_ls_xx_zz["d"].unique())
drop_ls = {s: {d: [1e-4] for d in all_ds} for s in df_ls_xx_zz["sub_experiment"].unique()}
plot_subexps(df_ls_xx_zz,
             "LS CNOT XX-ZZ — Sub-experiments",
             RESULTS / "fig1_cnot_ls_subexp_xx_zz.png",
             drop_p=drop_ls,
             x_min_override=3e-4)

# ── CNOT transversal ────────────────────────────────────────────────────────
df_ct = pd.read_csv(RESULTS / "fig1_cnot_trans_raw.csv")
plot_subexps(df_ct,
             "Transversal CNOT — Sub-experiments",
             RESULTS / "fig1_cnot_trans_subexp.png",
             drop_p={"ZZ_ZZ": {3: [1e-4], 5: [1e-4]}})

# ── H gate: average ZtoX and XtoZ, single panel ─────────────────────────────
df_h = pd.read_csv(RESULTS / "fig1_h_raw.csv")
df_h_avg = (df_h.groupby(["d", "p"])["logical_error_rate"]
            .mean().reset_index())
df_h_avg["sub_experiment"] = "H"
plot_subexps(df_h_avg,
             "Transversal H",
             RESULTS / "fig1_h_subexp.png",
             drop_p={"H": {3: [1e-4], 5: [1e-4]}},
             x_min_override=3e-4,
             subplot_title_override={"H": ""})

# ── S gate: per-round conversion 1-(1-LER)^(1/2) ────────────────────────────
df_s = pd.read_csv(RESULTS / "fig1_s_raw.csv")
df_s = df_s.copy()
df_s["logical_error_rate"] = 1 - (1 - df_s["logical_error_rate"]) ** 0.5
plot_subexps(df_s,
             "Transversal S",
             RESULTS / "fig1_s_subexp.png",
             drop_p={"S_roundtrip": {3: [1e-4], 5: [1e-4]}},
             x_min_override=3e-4,
             subplot_title_override={"S_roundtrip": ""})

print("Done.")
