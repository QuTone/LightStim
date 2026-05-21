"""
Bell-teleportation circuit benchmark: LER vs PER.
3 subplots: TG | LS-XX | LS-ZZ
Each subplot: d=3,5,7 (color) × state=Z/X (solid/dashed)

Usage:
    venv/bin/python -m eval.logical_circuit_benchmark.bell-teleportation.plot_bell_tele
    # or from project root:
    venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_bell_tele.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[3]))  # project root

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

RESULTS = Path(__file__).parent / "results"
OUTPUT  = RESULTS / "fig_bell_tele.png"

# ── Style constants ───────────────────────────────────────────────────────────
FS_TITLE  = 9
FS_LABEL  = 9
FS_TICK   = 8
FS_LEGEND = 9
LW        = 1.6
MS        = 6

LINESTYLE = {"Z": "-", "X": "--"}
MARKER    = {"Z": "o", "X": "s"}

DATASETS = [
    ("TG",     "tg_results.csv",    "Transversal Gate"),
    ("LS-XX",  "ls_xx_results.csv", "LS-XX"),
    ("LS-ZZ",  "ls_zz_results.csv", "LS-ZZ"),
]

# ── Load data ─────────────────────────────────────────────────────────────────
dfs = {key: pd.read_csv(RESULTS / fname) for key, fname, _ in DATASETS}

# ── Legend proxies ────────────────────────────────────────────────────────────
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

# ── One independent figure per protocol ──────────────────────────────────────
out_names = {"TG": "fig_bell_tele_tg.png",
             "LS-XX": "fig_bell_tele_ls_xx.png",
             "LS-ZZ": "fig_bell_tele_ls_zz.png"}

for key, _, title in DATASETS:
    df = dfs[key]
    fig, ax = plt.subplots(figsize=(2.5, 4))

    for d in [3, 5, 7]:
        color = PALETTE_DISTANCE[d]
        for state in ["Z", "X"]:
            sub = df[(df["d"] == d) & (df["state"] == state)].sort_values("p")
            if sub.empty:
                continue
            ax.loglog(sub["p"], sub["logical_error_rate"],
                      color=color,
                      linestyle=LINESTYLE[state],
                      marker=MARKER[state],
                      lw=LW, ms=MS, markeredgecolor="none")

    ax.set_xlabel("$p$", fontsize=FS_LABEL)
    ax.set_ylabel("LER", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    bold_ticks(ax)

    ax.legend(handles=dist_proxy + state_proxy,
              fontsize=FS_LEGEND,
              loc="lower right",
              frameon=True)

    out = RESULTS / out_names[key]
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
