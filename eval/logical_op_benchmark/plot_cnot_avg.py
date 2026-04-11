"""
Average CNOT LER across sub-experiments and plot.

Averaging formula (per d, p):
    avg = (ZZ_ZZ + ZX_ZX + (XZ_XX + XZ_ZZ)/2 + XX_XX) / 4

Outputs:
    fig_cnot_ls.png    — 2 subplots: ZZ-XX | XX-ZZ protocols
    fig1_cnot_trans.png — 1 subplot: transversal CNOT

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
LW, MS = 1.6, 6
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 10, 10, 10, 10


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


def plot_avg_ax(ax, avg_df, distances=(3, 5, 7)):
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
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", ls="--", alpha=0.4)
    bold_ticks(ax)


def distance_legend(distances=(3, 5, 7)):
    return [
        Line2D([], [], color=PALETTE_DISTANCE[d], marker=MARKERS[d],
               lw=LW, ms=MS, markeredgecolor="none", label=f"$d={d}$")
        for d in distances
    ]


# ── Fig 1: LS CNOT — two protocols ──────────────────────────────────────────
df_zz_xx = pd.read_csv(RESULTS / "fig1_cnot_ls_zz_xx_raw.csv")
df_xx_zz = pd.read_csv(RESULTS / "fig1_cnot_ls_xx_zz_raw.csv")

avg_zz_xx = compute_avg(df_zz_xx)
avg_xx_zz = compute_avg(df_xx_zz)

fig, axes = plt.subplots(1, 2, figsize=(4.8, 3.4),
                         sharey=True, constrained_layout=True)

for ax, avg_df, subtitle in zip(axes, [avg_zz_xx, avg_xx_zz],
                                ["LS CNOT (ZZ→XX)", "LS CNOT (XX→ZZ)"]):
    plot_avg_ax(ax, avg_df)
    ax.set_title(subtitle, fontsize=FS_TITLE)

axes[0].set_ylabel("LER", fontsize=FS_LABEL)

axes[-1].legend(
    handles=distance_legend(),
    fontsize=FS_LEGEND,
    loc="lower right",
    frameon=True,
)

out_ls = RESULTS / "fig1_cnot_ls.png"
fig.savefig(out_ls, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_ls}")

# ── Fig 2: Transversal CNOT — single panel ───────────────────────────────────
df_trans = pd.read_csv(RESULTS / "fig1_cnot_trans_raw.csv")
avg_trans = compute_avg(df_trans)

fig, ax = plt.subplots(1, 1, figsize=(2.6, 3.4), constrained_layout=True)
plot_avg_ax(ax, avg_trans)
ax.set_ylabel("LER", fontsize=FS_LABEL)
fig.suptitle("Transversal CNOT", fontweight="bold", fontsize=FS_TITLE + 1)

ax.legend(
    handles=distance_legend(),
    fontsize=FS_LEGEND,
    loc="lower right",
    frameon=True,
)

out_trans = RESULTS / "fig1_cnot_trans.png"
fig.savefig(out_trans, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_trans}")
