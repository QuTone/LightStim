"""
Plot memory benchmark results: logical error rate vs physical error rate.

Reads any CSV produced by run_memory.py.
One line per (code, distance). To keep the legend compact when many codes ×
distances are present, the figure uses two small legends instead of one entry
per curve:

    * color    encodes distance   (d=3, d=5, ...)
    * linestyle + marker encodes the code  (Rotated SC / Unrotated SC / Toric)

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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from lightstim.plot.styles import (
    apply_paper_style, bold_ticks, PALETTE_DISTANCE,
    CODE_LINESTYLES, CODE_MARKERS, CODE_LABELS,
)

apply_paper_style()


def _ordered_codes(df: pd.DataFrame) -> list:
    """Codes in canonical order (known first), then any extras alphabetically."""
    present = set(df["code"].unique())
    ordered = [c for c in CODE_LABELS if c in present]
    ordered += [c for c in sorted(present) if c not in ordered]
    return ordered


def plot_ler_vs_p(df: pd.DataFrame, ax: plt.Axes, title: str = "") -> None:
    """One curve per (code, distance): color = distance, linestyle+marker = code."""
    codes     = _ordered_codes(df)
    distances = sorted(df["distance"].unique())

    for code in codes:
        ls = CODE_LINESTYLES.get(code, "-")
        mk = CODE_MARKERS.get(code, "o")
        for d in distances:
            sub = df[(df["code"] == code) & (df["distance"] == d)].sort_values("p")
            if sub.empty:
                continue
            ax.plot(
                sub["p"], sub["logical_error_rate"],
                ls=ls, marker=mk, color=PALETTE_DISTANCE.get(int(d), "0.4"),
                lw=2, ms=7, markeredgecolor="k", markeredgewidth=0.4,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate $p$", fontsize=13)
    ax.set_ylabel("Logical error rate", fontsize=13)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    bold_ticks(ax)

    # ── Two compact legends ─────────────────────────────────────────────────
    # Legend 1: code → linestyle + marker (neutral color).
    code_handles = [
        Line2D([0], [0], color="0.2", lw=2,
               ls=CODE_LINESTYLES.get(c, "-"), marker=CODE_MARKERS.get(c, "o"),
               ms=7, markeredgecolor="k", markeredgewidth=0.4,
               label=CODE_LABELS.get(c, c))
        for c in codes
    ]
    # Legend 2: distance → color (marker only, no line).
    dist_handles = [
        Line2D([0], [0], color=PALETTE_DISTANCE.get(int(d), "0.4"),
               lw=0, marker="o", ms=8, markeredgecolor="k", markeredgewidth=0.4,
               label=f"d={d}")
        for d in distances
    ]
    leg_codes = ax.legend(handles=code_handles, loc="upper left",
                          fontsize=10, framealpha=0.9)
    ax.add_artist(leg_codes)
    ax.legend(handles=dist_handles, loc="lower right",
              fontsize=10, framealpha=0.9, ncol=1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="Input CSV file(s) from run_memory.py")
    ap.add_argument("--output", default=None,
                    help="Output path (extension sets format: .pdf, .svg, .png)")
    ap.add_argument("--codes", nargs="*", default=None,
                    help="Filter to specific code name(s)")
    ap.add_argument("--distances", nargs="*", type=int, default=None,
                    help="Filter to specific distance(s)")
    ap.add_argument("--p-min", type=float, default=None,
                    help="Drop points with p < P_MIN")
    ap.add_argument("--p-max", type=float, default=None,
                    help="Drop points with p > P_MAX")
    ap.add_argument("--per-logical", action="store_true",
                    help="Divide LER by k (number of logical qubits) so multi-logical "
                         "codes like toric (k=2) are shown per logical qubit.")
    ap.add_argument("--title", default=None, help="Plot title")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.inputs], ignore_index=True)

    if args.codes:
        df = df[df["code"].isin(args.codes)]
    if args.distances:
        df = df[df["distance"].isin(args.distances)]
    if args.p_min is not None:
        df = df[df["p"] >= args.p_min * (1 - 1e-9)]
    if args.p_max is not None:
        df = df[df["p"] <= args.p_max * (1 + 1e-9)]

    if df.empty:
        print("No data after filtering — nothing to plot.")
        return

    if args.per_logical and "k" in df.columns:
        df = df.copy()
        df["logical_error_rate"] = df["logical_error_rate"] / df["k"]

    title = args.title or (", ".join(sorted(df["code"].unique())))
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    plot_ler_vs_p(df, ax, title=title)
    if args.per_logical:
        ax.set_ylabel("Logical error rate per logical qubit", fontsize=13)

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
