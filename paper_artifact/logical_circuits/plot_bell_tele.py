"""
Plot Bell-state teleportation LER vs PER for three protocols.

Reads:   precomputed/bell_tele_{tg,ls_zz,ls_xx}.csv
Outputs: results/bell_tele_{tg,ls_zz,ls_xx}.png
         results/bell_tele.png  (3-panel combined)

Usage (from repo root):
    venv/bin/python paper_artifact/logical_circuits/plot_bell_tele.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

HERE       = Path(__file__).parent
PRECOMP    = HERE / "precomputed"
OUT_DIR    = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

# Style
LW, MS       = 1.6, 6
FS_TITLE     = 9
FS_LABEL     = 9
FS_TICK      = 8
FS_LEGEND    = 9
LINESTYLE    = {"Z": "-",  "X": "--"}
MARKER       = {"Z": "o",  "X": "s"}

DATASETS = [
    ("TG",     "bell_tele_tg.csv",     "Transversal Gate"),
    ("LS-ZZ",  "bell_tele_ls_zz.csv",  "LS-ZZ"),
    ("LS-XX",  "bell_tele_ls_xx.csv",  "LS-XX"),
]


def plot_one(ax, df, title):
    for d in [3, 5, 7]:
        color = PALETTE_DISTANCE[d]
        for state in ["Z", "X"]:
            sub = df[(df["d"] == d) & (df["state"] == state)].sort_values("p")
            if sub.empty:
                continue
            ax.loglog(sub["p"], sub["logical_error_rate"],
                      color=color, linestyle=LINESTYLE[state],
                      marker=MARKER[state], lw=LW, ms=MS, markeredgecolor="none")
    ax.set_xlabel("$p$", fontsize=FS_LABEL)
    ax.set_ylabel("LER",  fontsize=FS_LABEL)
    ax.set_title(title,   fontsize=FS_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    bold_ticks(ax)


dist_proxy = [
    Line2D([], [], color=PALETTE_DISTANCE[d], ls="-", lw=LW,
           marker="o", ms=MS, markeredgecolor="none", label=f"d={d}")
    for d in [3, 5, 7]
]
state_proxy = [
    Line2D([], [], color="black", ls="-",  lw=LW, marker="o", ms=MS,
           markeredgecolor="none", label="Z state"),
    Line2D([], [], color="black", ls="--", lw=LW, marker="s", ms=MS,
           markeredgecolor="none", label="X state"),
]

dfs = {}
for key, fname, _ in DATASETS:
    path = PRECOMP / fname
    dfs[key] = pd.read_csv(path)
    print(f"Loaded {fname}: {len(dfs[key])} rows")

# Individual panels
for key, _, title in DATASETS:
    fig, ax = plt.subplots(figsize=(2.5, 4))
    plot_one(ax, dfs[key], title)
    ax.legend(handles=dist_proxy + state_proxy, fontsize=FS_LEGEND,
              loc="lower right", frameon=True)
    out = OUT_DIR / f"bell_tele_{key.lower().replace('-', '_')}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# Combined 3-panel figure (matching paper Bell-state-tele.png)
fig, axes = plt.subplots(1, 3, figsize=(7.5, 4), sharey=False)
for ax, (key, _, title) in zip(axes, DATASETS):
    plot_one(ax, dfs[key], title)

axes[0].legend(handles=dist_proxy + state_proxy, fontsize=FS_LEGEND - 1,
               loc="lower right", frameon=True)

fig.tight_layout()
out_combined = OUT_DIR / "bell_tele.png"
fig.savefig(out_combined, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_combined}")
