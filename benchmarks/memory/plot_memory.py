"""
Plot memory benchmark results: logical error rate vs physical error rate.

Reads any CSV produced by run_memory.py.
Groups curves by (code, distance) — one line per group.

Usage
-----
    # Single file:
    venv/bin/python benchmarks/memory/plot_memory.py results/rotated_sc_pymatching.csv

    # Multiple files merged:
    venv/bin/python benchmarks/memory/plot_memory.py results/*.csv

    # Filter codes/distances, custom output:
    venv/bin/python benchmarks/memory/plot_memory.py results/surface_pymatching.csv \\
        --codes rotated_sc --distances 3 5 7 \\
        --output results/rotated_sc.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE_DISTANCE

apply_paper_style()

_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


def plot_ler_vs_p(df: pd.DataFrame, ax: plt.Axes, title: str = "") -> None:
    """Plot one curve per (code, distance) group."""
    groups = df.groupby(["code", "distance"], sort=True)
    for i, ((code, d), sub) in enumerate(groups):
        sub = sub.sort_values("p")
        color  = PALETTE_DISTANCE.get(int(d), f"C{i % 10}")
        marker = _MARKERS[i % len(_MARKERS)]
        label  = f"{code} d={d}"
        ax.plot(
            sub["p"], sub["logical_error_rate"],
            marker=marker, color=color, lw=2, ms=7,
            markeredgecolor="k", markeredgewidth=0.4,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate $p$", fontsize=13)
    ax.set_ylabel("Logical error rate", fontsize=13)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.85, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    bold_ticks(ax)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="Input CSV file(s) from run_memory.py")
    ap.add_argument("--output", default=None,
                    help="Output PNG path (default: same dir as first input)")
    ap.add_argument("--codes", nargs="*", default=None,
                    help="Filter to specific code name(s)")
    ap.add_argument("--distances", nargs="*", type=int, default=None,
                    help="Filter to specific distance(s)")
    ap.add_argument("--title", default=None, help="Plot title")
    args = ap.parse_args()

    dfs = [pd.read_csv(p) for p in args.inputs]
    df  = pd.concat(dfs, ignore_index=True)

    if args.codes:
        df = df[df["code"].isin(args.codes)]
    if args.distances:
        df = df[df["distance"].isin(args.distances)]

    if df.empty:
        print("No data after filtering — nothing to plot.")
        return

    title = args.title or (", ".join(sorted(df["code"].unique())))
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    plot_ler_vs_p(df, ax, title=title)

    if args.output:
        out = Path(args.output)
    else:
        out = Path(args.inputs[0]).parent / "memory_ler_vs_p.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
