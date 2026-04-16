"""
State injection single-panel figures for corner injection (rotated SC).

Generates 4 independent figures:
  fig2_corner_states_d3.png  — d=3, X/Y/Z states, corner, full_PS, 2 rounds
  fig2_corner_states_d5.png  — d=5, X/Y/Z states, corner, full_PS, 2 rounds
  fig2_corner_states_d7.png  — d=7, X/Y/Z states, corner, full_PS, 2 rounds
  fig2_corner_modes_d7.png   — d=7, 3 modes, corner, Z state, 2 rounds

Format: single-panel, dual-y (LER log-log left, PS Rate semilog-x right).
Size: 3.2 × 3.4 inches — consistent with other standalone panels (H, S, Memory).

Usage:
    venv/bin/python -m eval.logical_op_benchmark.plot_state_injection_singles
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from src.plot.styles import apply_paper_style, PALETTE, bold_ticks

apply_paper_style()

CSV = Path("eval/logical_op_benchmark/state_injection/results_rotated/state_injection_eval.csv")
OUT = Path("eval/logical_op_benchmark/results")

# ── Font / line sizes ────────────────────────────────────────────────────────
FS_TITLE  = 16
FS_LABEL  = 12
FS_TICK   = 12
FS_LEGEND = 11
LW        = 1.8
MS        = 6
FIG_W     = 4.5
FIG_H     = 4.2

# ── Color maps ────────────────────────────────────────────────────────────────
STATE_COLOR  = {"Z": "#2166ac", "X": "#b22222", "Y": "#1b9e77"}
STATE_LABEL  = {"Z": r"$|Z\rangle$", "X": r"$|X\rangle$", "Y": r"$|Y\rangle$"}
STATE_MARKER = {"Z": "o", "X": "s", "Y": "^"}

MODE_COLOR   = {"full_postselection": PALETTE[0], "full_qec": PALETTE[1], "hybrid": PALETTE[2]}
MODE_LABEL   = {"full_postselection": "Full PS", "full_qec": "Full QEC", "hybrid": "Hybrid"}
MODE_MARKER  = {"full_postselection": "o", "full_qec": "s", "hybrid": "^"}

df_all = pd.read_csv(CSV)
df_all = df_all[df_all["rounds"] == 2]


def make_single_panel(df_sub, group_key, group_vals,
                      group_color, group_label, group_marker,
                      title, outname):
    """Single-panel dual-y figure: LER (log-log, left) + PS Rate (semilog-x, right)."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax2 = ax.twinx()

    legend_handles = []
    legend_labels  = []

    for gval in group_vals:
        rows = df_sub[df_sub[group_key] == gval].sort_values("p")
        if rows.empty:
            continue

        color  = group_color[gval]
        label  = group_label[gval]
        marker = group_marker[gval]

        p_vals  = rows["p"].values
        ler     = rows["logical_error_rate"].values
        ps_rate = rows["post_selection_rate"].values

        # LER — solid lines, left axis (log-log)
        line, = ax.loglog(p_vals, ler,
                          marker=marker, color=color,
                          lw=LW, ms=MS, markeredgecolor="none",
                          linestyle="-", zorder=3)

        # PS rate — dashed, right axis (semilog-x, linear y)
        ax2.semilogx(p_vals, ps_rate,
                     marker=marker, color=color,
                     lw=LW * 0.85, ms=MS * 0.7, markeredgecolor="none",
                     linestyle="--", alpha=0.75, zorder=2)

        legend_handles.append(line)
        legend_labels.append(label)

    # Left axis
    ax.set_xlabel("$p$", fontsize=FS_LABEL)
    ax.set_ylabel("LER", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    bold_ticks(ax)

    # Right axis
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("PS Rate", fontsize=FS_LABEL, labelpad=12)
    ax2.tick_params(labelsize=FS_TICK)
    bold_ticks(ax2)

    # Legend: group lines + line-style proxies for LER / PS Rate
    proxy_ler = Line2D([], [], color="k", ls="-",  lw=LW, label="LER")
    proxy_ps  = Line2D([], [], color="k", ls="--", lw=LW * 0.85, alpha=0.75, label="PS Rate")

    all_handles = legend_handles + [proxy_ler, proxy_ps]
    all_labels  = legend_labels  + ["LER", "PS Rate"]

    ax.legend(all_handles, all_labels,
              fontsize=FS_LEGEND,
              title_fontsize=FS_LEGEND,
              loc="upper left",
              frameon=True,
              framealpha=0.4)

    out_path = OUT / outname
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Corner Injection — X/Y/Z states, each distance separately ────────────────
df_corner_states = df_all[
    (df_all["injection_protocol"] == "corner") &
    (df_all["post_select_mode"] == "full_postselection")
].copy()

for d in [3, 5, 7]:
    make_single_panel(
        df_corner_states[df_corner_states["d"] == d],
        group_key="inject_state",
        group_vals=["Z", "X", "Y"],
        group_color=STATE_COLOR,
        group_label=STATE_LABEL,
        group_marker=STATE_MARKER,
        title=f"Corner Injection — $d={d}$",
        outname=f"fig2_corner_states_d{d}.png",
    )

# ── Middle Injection — X/Y/Z states, each distance separately ────────────────
df_middle_states = df_all[
    (df_all["injection_protocol"] == "middle") &
    (df_all["post_select_mode"] == "full_postselection")
].copy()

for d in [3, 5, 7]:
    make_single_panel(
        df_middle_states[df_middle_states["d"] == d],
        group_key="inject_state",
        group_vals=["Z", "X", "Y"],
        group_color=STATE_COLOR,
        group_label=STATE_LABEL,
        group_marker=STATE_MARKER,
        title=f"Middle Injection — $d={d}$",
        outname=f"fig2_middle_states_d{d}.png",
    )

# ── Corner Injection — Z state, 3 modes, d=7 only ────────────────────────────
df_corner_modes = df_all[
    (df_all["injection_protocol"] == "corner") &
    (df_all["inject_state"] == "Z") &
    (df_all["d"] == 7)
].copy()

make_single_panel(
    df_corner_modes,
    group_key="post_select_mode",
    group_vals=["full_postselection", "full_qec", "hybrid"],
    group_color=MODE_COLOR,
    group_label=MODE_LABEL,
    group_marker=MODE_MARKER,
    title=r"Corner Injection — $|Z\rangle$, $d=7$",
    outname="fig2_corner_modes_d7.png",
)

print("Done. Generated 7 figures:")
print("  fig2_corner_states_d3/5/7.png")
print("  fig2_middle_states_d3/5/7.png")
print("  fig2_corner_modes_d7.png")
