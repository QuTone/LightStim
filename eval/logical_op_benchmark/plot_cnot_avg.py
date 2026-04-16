"""
Average CNOT LER across sub-experiments and plot.

Averaging formula (per d, p):
    avg = (ZZ_ZZ + ZX_ZX + (XZ_XX + XZ_ZZ)/2 + XX_XX) / 4

Outputs (all independent single-panel figures):
    fig1_cnot_ls_zz_xx.png  — LS CNOT ZZ→XX (averaged)
    fig1_cnot_ls_xx_zz.png  — LS CNOT XX→ZZ (averaged)
    fig1_cnot_trans.png      — Transversal CNOT (averaged)

Usage:
    venv/bin/python -m eval.logical_op_benchmark.plot_cnot_avg
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from src.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

RESULTS = Path("eval/logical_op_benchmark/results")
MARKERS = {3: "o", 5: "s", 7: "^"}
LW, MS = 2.2, 8
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 16, 12, 12, 11
FIG_W, FIG_H = 4.5, 4.2   # single-panel size (inches)


def compute_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average LER across the 5 CNOT sub-experiments:
        avg = (ZZ_ZZ + ZX_ZX + (XZ_XX + XZ_ZZ)/2 + XX_XX) / 4
    Returns DataFrame with columns [d, p, logical_error_rate].
    """
    required = {"ZZ_ZZ", "ZX_ZX", "XZ_XX", "XZ_ZZ", "XX_XX"}
    rows = []
    for (d, p), grp in df.groupby(["d", "p"]):
        ler = grp.set_index("sub_experiment")["logical_error_rate"]
        if not required.issubset(set(ler.index)):
            continue  # skip incomplete (d, p) points
        avg = (
            ler["ZZ_ZZ"]
            + ler["ZX_ZX"]
            + (ler["XZ_XX"] + ler["XZ_ZZ"]) / 2
            + ler["XX_XX"]
        ) / 4
        rows.append({"d": int(d), "p": float(p), "logical_error_rate": float(avg)})
    return pd.DataFrame(rows).sort_values(["d", "p"])


def plot_single(avg_df, title, out_path, distances=(3, 5, 7)):
    """Independent single-panel averaged CNOT figure."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for d in distances:
        sub = avg_df[avg_df["d"] == d].sort_values("p")
        if sub.empty:
            continue
        ax.loglog(
            sub["p"], sub["logical_error_rate"],
            color=PALETTE_DISTANCE[d],
            marker=MARKERS[d],
            lw=LW, ms=MS,
            markeredgecolor="none",
        )

    ax.set_xlim(left=3e-4)
    ax.set_xlabel("$p$", fontsize=FS_LABEL)
    ax.set_ylabel("LER", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", ls="--", alpha=0.4)
    bold_ticks(ax)

    legend_handles = [
        Line2D([], [], color=PALETTE_DISTANCE[d], marker=MARKERS[d],
               lw=LW, ms=MS, markeredgecolor="none", label=f"$d={d}$")
        for d in distances
    ]
    ax.legend(handles=legend_handles, fontsize=FS_LEGEND,
              loc="lower right", frameon=True)

    ax.set_title(title, fontweight="bold", fontsize=FS_TITLE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Fig 1: LS CNOT ZZ→XX — independent single panel ────────────────────────
df_zz_xx = pd.read_csv(RESULTS / "fig1_cnot_ls_zz_xx_raw.csv")
avg_zz_xx = compute_avg(df_zz_xx)
plot_single(avg_zz_xx, "LS CNOT (ZZ→XX)",
            RESULTS / "fig1_cnot_ls_zz_xx.png")

# ── Fig 2: LS CNOT XX→ZZ — independent single panel ────────────────────────
df_xx_zz = pd.read_csv(RESULTS / "fig1_cnot_ls_xx_zz_raw.csv")
avg_xx_zz = compute_avg(df_xx_zz)
plot_single(avg_xx_zz, "LS CNOT (XX→ZZ)",
            RESULTS / "fig1_cnot_ls_xx_zz.png")

# ── Fig 3: Transversal CNOT — independent single panel ──────────────────────
df_trans = pd.read_csv(RESULTS / "fig1_cnot_trans_raw.csv")
avg_trans = compute_avg(df_trans)
plot_single(avg_trans, "Transversal CNOT",
            RESULTS / "fig1_cnot_trans.png")
