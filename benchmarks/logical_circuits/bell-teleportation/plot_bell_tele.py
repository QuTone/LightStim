"""
Bell-teleportation circuit benchmark: LER vs PER.
One figure per protocol (TG | LS-ZZ | LS-XX).
Each figure: d=3,5,7 (color) × state=Z/X (solid/dashed)

Data:  ../results/bell_tele_results.csv
       Generate with:
           PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/run_logical_circuits.py \\
               --experiment bell_tele --distances 3 5 7 --p-values 5e-4 1e-3 2e-3 5e-3

Usage:
    PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/bell-teleportation/plot_bell_tele.py
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

CSV     = Path(__file__).resolve().parents[1] / "results" / "bell_tele_results.csv"
OUT_DIR = CSV.parent
OUT_DIR.mkdir(exist_ok=True)

# ── Style constants ───────────────────────────────────────────────────────────
FS_TITLE  = 9
FS_LABEL  = 9
FS_TICK   = 8
FS_LEGEND = 9
LW        = 1.6
MS        = 6

LINESTYLE = {"Z": "-", "X": "--"}
MARKER    = {"Z": "o", "X": "s"}

PROTOCOLS = [
    ("tg",    "Transversal Gate",  "fig_bell_tele_tg.png"),
    ("ls_zz", "LS-ZZ",             "fig_bell_tele_ls_zz.png"),
    ("ls_xx", "LS-XX",             "fig_bell_tele_ls_xx.png"),
]

# ── Load data ─────────────────────────────────────────────────────────────────
df_all = pd.read_csv(CSV)

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

# ── One figure per protocol ───────────────────────────────────────────────────
for proto_key, title, out_fname in PROTOCOLS:
    df = df_all[df_all["protocol"] == proto_key]
    if df.empty:
        print(f"No data for protocol={proto_key!r}, skipping.")
        continue

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

    out = OUT_DIR / out_fname
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
