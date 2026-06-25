"""
S-gate teleportation benchmark plot: LER vs PER.

Data:
    benchmarks/logical_circuits/results/s_gate_tele_results.csv

Usage:
    PYTHONPATH=. venv/bin/python benchmarks/logical_circuits/s-gate-teleportation/plot_s_gate_tele.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from lightstim.plot.styles import PALETTE_DISTANCE


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 1.0,
})

CSV = Path(__file__).resolve().parents[1] / "results" / "s_gate_tele_results.csv"
OUT = CSV.parent / "fig_s_gate_tele.png"

METHOD_LABEL = {
    "ZZ": "LS-ZZ",
    "cnot_trans": "Transversal CNOT",
}
METHOD_STYLE = {
    "ZZ": ("-", "o"),
    "cnot_trans": ("--", "s"),
}


def main():
    if not CSV.exists():
        raise FileNotFoundError(
            f"Missing {CSV}. Run the s_gate_tele benchmark before plotting."
        )

    df = pd.read_csv(CSV)
    df = df[
        (df["experiment"] == "s_gate_tele")
        & (df["code"] == "unrotated_sc")
        & (df["state_prep"] == "logical_gate")
    ].copy()
    if df.empty:
        raise ValueError("No unrotated logical_gate s_gate_tele rows found.")

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    for method in ["ZZ", "cnot_trans"]:
        ls, marker = METHOD_STYLE[method]
        for d in sorted(df["d"].unique()):
            sub = df[(df["method"] == method) & (df["d"] == d)].sort_values("p")
            if sub.empty:
                continue
            y = sub["logical_error_rate"].where(
                sub["logical_error_rate"] > 0,
                0.5 / sub["shots"],
            )
            ax.loglog(
                sub["p"],
                y,
                color=PALETTE_DISTANCE.get(int(d), "black"),
                linestyle=ls,
                marker=marker,
                linewidth=1.6,
                markersize=5.5,
                markeredgecolor="none",
            )

    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate")
    ax.set_title("S-gate teleportation")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

    dist_proxy = [
        Line2D(
            [],
            [],
            color=PALETTE_DISTANCE.get(int(d), "black"),
            linestyle="-",
            marker="o",
            markersize=5.5,
            markeredgecolor="none",
            label=f"d={int(d)}",
        )
        for d in sorted(df["d"].unique())
    ]
    method_proxy = [
        Line2D(
            [],
            [],
            color="black",
            linestyle=METHOD_STYLE[m][0],
            marker=METHOD_STYLE[m][1],
            markersize=5.5,
            markeredgecolor="none",
            label=METHOD_LABEL[m],
        )
        for m in ["ZZ", "cnot_trans"]
    ]
    leg1 = ax.legend(handles=dist_proxy, loc="lower right", frameon=False, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=method_proxy, loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
