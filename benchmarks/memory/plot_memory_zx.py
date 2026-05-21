"""
Plot Memory Z and Memory X separately for unrotated SC.
Same format as fig1_memory_baseline.png.
Output: benchmarks/logical_ops/results/fig1_memory_Z.png / fig1_memory_X.png

Usage:
    venv/bin/python -m eval.memory_benchmark.plot_memory_zx
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

MEMORY_CSV = Path("benchmarks/memory/results/fig1_surface_codes.csv")
OUT        = Path("benchmarks/logical_ops/results")
MARKERS    = {3: "o", 5: "s", 7: "^"}

FS_TITLE  = 12
FS_LABEL  = 12
FS_TICK   = 11
FS_LEGEND = 10
LW        = 2.2
MS        = 8

df_all = pd.read_csv(MEMORY_CSV)
df_all = df_all[df_all["code"] == "unrotated_sc"].copy()


def make_fig(basis, title, outname):
    sub_all = df_all[df_all["basis"] == basis]

    fig, ax = plt.subplots(figsize=(3.0, 4.2), constrained_layout=True)
    handles, labels = [], []

    for d in [3, 5, 7]:
        sub = sub_all[sub_all["distance"] == d].sort_values("p")
        sub = sub[(sub["p"] >= 3e-4) & (sub["p"] <= 1e-2)]
        if sub.empty:
            continue

        color = PALETTE_DISTANCE[d]
        line, = ax.loglog(sub["p"], sub["logical_error_rate"],
                          marker=MARKERS[d], color=color,
                          lw=LW, ms=MS, markeredgecolor="none",
                          label=f"d={d}")
        handles.append(line)
        labels.append(f"d={d}")

    ax.set_xlim(3e-4, 2e-2)
    ax.set_xlabel("$p$", fontsize=FS_LABEL)
    ax.set_ylabel("LER", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    bold_ticks(ax)

    ax.legend(handles, labels,
              title="Distance",
              title_fontsize=FS_LEGEND,
              fontsize=FS_LEGEND,
              loc="center left",
              bbox_to_anchor=(1.02, 0.5),
              frameon=True)

    out_path = OUT / outname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


make_fig("Z", "Memory Z", "fig1_memory_Z.png")
make_fig("X", "Memory X", "fig1_memory_X.png")
