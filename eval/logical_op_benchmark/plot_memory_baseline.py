"""
Plot Unrotated SC Z/X memory LER / 2 at matching p values.
Size: 3.0 x 4.2 inches, same style as H/S figures.
Output: eval/logical_op_benchmark/results/fig1_memory_baseline.png

Usage:
    venv/bin/python -m eval.logical_op_benchmark.plot_memory_baseline
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

MEMORY_CSV = Path("eval/memory_benchmark/results/fig1_surface_codes.csv")
OUTPUT     = Path("eval/logical_op_benchmark/results/fig1_memory_baseline.png")
MARKERS    = {3: "o", 5: "s", 7: "^"}

df = pd.read_csv(MEMORY_CSV)
df = df[df["code"] == "unrotated_sc"].copy()
# Average Z and X basis LER per (distance, p)
df = df.groupby(["distance", "p"])["logical_error_rate"].mean().reset_index()

# Single-panel font sizes — match fig1_s_subexp.png / fig1_h_subexp.png
FS_TITLE  = 12
FS_LABEL  = 10
FS_TICK   = 11
FS_LEGEND = 10
LW        = 2.2
MS        = 8

fig, ax = plt.subplots(figsize=(2.7, 3.4), constrained_layout=True)

handles, labels = [], []

for d in [3, 5, 7]:
    sub = df[df["distance"] == d].sort_values("p")
    sub = sub[(sub["p"] >= 3e-4) & (sub["p"] <= 1e-2)]
    if sub.empty:
        continue

    p_vals   = sub["p"].values
    ler_div2 = sub["logical_error_rate"].values  # already averaged over Z/X

    color = PALETTE_DISTANCE[d]
    line, = ax.loglog(p_vals, ler_div2,
                      marker=MARKERS[d], color=color,
                      lw=LW, ms=MS, markeredgecolor="none",
                      label=f"d={d}")
    handles.append(line)
    labels.append(f"d={d}")

ax.set_xlim(3e-4, 2e-2)
ax.set_xlabel("$p$", fontsize=FS_LABEL)
ax.set_ylabel("LER", fontsize=FS_LABEL)
ax.set_title("Memory", fontsize=FS_TITLE)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, which="major", ls="--", alpha=0.5)
bold_ticks(ax)

ax.legend(handles, labels,
          title="Distance",
          title_fontsize=FS_LEGEND,
          fontsize=FS_LEGEND,
          loc="lower right",
          frameon=True)

fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
