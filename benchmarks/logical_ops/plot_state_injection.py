"""
State injection benchmark figures for rotated surface code.
Format: unified with memory / logical_op figures (compact 3-panel, apply_paper_style).

fig2_corner_states.png:  corner injection, full_PS, 2 rounds, Z/X/Y states
fig2_middle_modes.png:   middle injection, Z state, 2 rounds, full_PS/full_qec/hybrid
fig2_middle_states.png:  middle injection, full_PS, 2 rounds, Z/X/Y states
fig2_corner_modes.png:   corner injection, Z state, 2 rounds, full_PS/full_qec/hybrid

Output: benchmarks/logical_ops/results/

Usage:
    venv/bin/python -m eval.logical_op_benchmark.plot_state_injection
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, PALETTE, bold_ticks

apply_paper_style()

CSV   = Path("benchmarks/logical_ops/state_injection/results_rotated/state_injection_eval.csv")
OUT   = Path("benchmarks/logical_ops/results")
DISTS = [3, 5, 7]

# Compact 3-panel font sizes — same as plot_logical_op.py compact mode
FS_TITLE  = 9
FS_LABEL  = 9
FS_TICK   = 8
FS_LEGEND = 8
LW        = 1.6
MS        = 6

# ── Color maps ────────────────────────────────────────────────────────────────
STATE_COLOR  = {"Z": "#2166ac", "X": "#b22222", "Y": "#1b9e77"}  # blue / red / green
STATE_LABEL  = {"Z": "$|Z\\rangle$", "X": "$|X\\rangle$", "Y": "$|Y\\rangle$"}
STATE_MARKER = {"Z": "o", "X": "s", "Y": "^"}

MODE_COLOR   = {"full_postselection": PALETTE[0], "full_qec": PALETTE[1], "hybrid": PALETTE[2]}
MODE_LABEL   = {"full_postselection": "Full PS", "full_qec": "Full QEC", "hybrid": "Hybrid"}
MODE_MARKER  = {"full_postselection": "o", "full_qec": "s", "hybrid": "^"}

df_all = pd.read_csv(CSV)
df_all = df_all[df_all["rounds"] == 2]


def make_fig(df, group_key, group_vals, group_color, group_label, group_marker,
             suptitle, outname, height=4.2):
    """3-panel (d=3,5,7) dual-y figure: left=LER (log), right=PS_rate (linear)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, height),
                             constrained_layout=True)

    legend_handles = []
    legend_labels  = []

    for i, (ax, d) in enumerate(zip(axes, DISTS)):
        sub = df[df["d"] == d]

        ax2 = ax.twinx()

        for gval in group_vals:
            rows = sub[sub[group_key] == gval].sort_values("p")
            if rows.empty:
                continue
            color  = group_color[gval]
            label  = group_label[gval]
            marker = group_marker[gval]

            p_vals  = rows["p"].values
            ler     = rows["logical_error_rate"].values
            ps_rate = rows["post_selection_rate"].values

            # LER — solid, left axis (log-log)
            line, = ax.loglog(p_vals, ler,
                              marker=marker, color=color,
                              lw=LW, ms=MS, markeredgecolor="none",
                              linestyle="-", zorder=3)

            # PS rate — dashed, right axis (log x, linear y)
            ax2.semilogx(p_vals, ps_rate,
                         marker=marker, color=color,
                         lw=LW * 0.85, ms=MS * 0.7, markeredgecolor="none",
                         linestyle="--", alpha=0.75, zorder=2)

            if i == 0:
                legend_handles.append(line)
                legend_labels.append(label)

        # Left axis styling
        ax.set_xlabel("$p$", fontsize=FS_LABEL)
        ax.set_title(f"$d={d}$", fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, which="major", ls="--", alpha=0.5)
        bold_ticks(ax)

        # Right axis styling
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(labelsize=FS_TICK)
        if i < len(DISTS) - 1:
            ax2.set_yticklabels([])
        else:
            ax2.set_ylabel("PS Rate", fontsize=FS_LABEL, labelpad=14)
            bold_ticks(ax2)

    axes[0].set_ylabel("LER", fontsize=FS_LABEL)

    # Legend: group colors + line-style proxies
    proxy_ler = Line2D([], [], color="k", ls="-",  lw=LW, label="LER")
    proxy_ps  = Line2D([], [], color="k", ls="--", lw=LW * 0.85, alpha=0.75, label="PS Rate")

    all_handles = legend_handles + [proxy_ler, proxy_ps]
    all_labels  = legend_labels  + ["LER", "PS Rate"]

    axes[-1].legend(all_handles, all_labels,
                    fontsize=FS_LEGEND,
                    title_fontsize=FS_LEGEND,
                    loc="upper left",
                    frameon=True,
                    framealpha=0.35)

    fig.suptitle(suptitle, fontweight="bold", fontsize=FS_TITLE + 1)
    out_path = OUT / outname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Fig 2a: corner, full_PS, 2 rounds, Z/X/Y ─────────────────────────────────
df_a = df_all[
    (df_all["injection_protocol"] == "corner") &
    (df_all["post_select_mode"] == "full_postselection")
].copy()

make_fig(
    df_a,
    group_key="inject_state",
    group_vals=["Z", "X", "Y"],
    group_color=STATE_COLOR,
    group_label=STATE_LABEL,
    group_marker=STATE_MARKER,
    suptitle="Corner Injection — Full PS, 2 Rounds",
    outname="fig2_corner_states.png",
)

# ── Fig 2b: middle, Z, 2 rounds, full_PS/full_qec/hybrid ─────────────────────
df_b = df_all[
    (df_all["injection_protocol"] == "middle") &
    (df_all["inject_state"] == "Z")
].copy()

make_fig(
    df_b,
    group_key="post_select_mode",
    group_vals=["full_postselection", "full_qec", "hybrid"],
    group_color=MODE_COLOR,
    group_label=MODE_LABEL,
    group_marker=MODE_MARKER,
    suptitle="Middle Injection — Z State, 2 Rounds",
    outname="fig2_middle_modes.png",
)

# ── Fig 2c: middle, full_PS, 2 rounds, Z/X/Y ─────────────────────────────────
df_c = df_all[
    (df_all["injection_protocol"] == "middle") &
    (df_all["post_select_mode"] == "full_postselection")
].copy()

make_fig(
    df_c,
    group_key="inject_state",
    group_vals=["Z", "X", "Y"],
    group_color=STATE_COLOR,
    group_label=STATE_LABEL,
    group_marker=STATE_MARKER,
    suptitle="Middle Injection — Full PS, 2 Rounds",
    outname="fig2_middle_states.png",
    height=3.4,
)

# ── Fig 2d: corner, Z, 2 rounds, full_PS/full_qec/hybrid ─────────────────────
df_d = df_all[
    (df_all["injection_protocol"] == "corner") &
    (df_all["inject_state"] == "Z")
].copy()

make_fig(
    df_d,
    group_key="post_select_mode",
    group_vals=["full_postselection", "full_qec", "hybrid"],
    group_color=MODE_COLOR,
    group_label=MODE_LABEL,
    group_marker=MODE_MARKER,
    suptitle="Corner Injection — Z State, 2 Rounds",
    outname="fig2_corner_modes.png",
    height=3.4,
)

print("Done.")
