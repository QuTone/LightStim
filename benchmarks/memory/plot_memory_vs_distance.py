"""
Plot memory benchmark results: logical error rate vs code distance.

Reads any CSV produced by run_memory.py. Fixes the physical error rate ``p``
(default 1e-3) and draws one curve per code, with distance on the x-axis.

Usage
-----
    # All codes in the CSV, fixed p = 1e-3:
    venv/bin/python benchmarks/memory/plot_memory_vs_distance.py \\
        handbook_result/surface_family.csv \\
        --p 1e-3 --output handbook_result/ler_vs_distance.pdf

    # Filter codes / custom title:
    venv/bin/python benchmarks/memory/plot_memory_vs_distance.py results/*.csv \\
        --codes rotated_sc unrotated_sc toric \\
        --p 1e-3 --title "Memory @ p=1e-3"
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

from lightstim.plot.styles import (
    apply_paper_style, bold_ticks, CODES, CODE_LABELS, CODE_MARKERS,
)

apply_paper_style()


def _closest_p(df: pd.DataFrame, p: float) -> float:
    """Snap requested p to the nearest value actually present in the data."""
    available = np.sort(df["p"].unique())
    return float(available[int(np.argmin(np.abs(available - p)))])


def plot_ler_vs_distance(df: pd.DataFrame, ax: plt.Axes, title: str = "") -> None:
    """Plot one curve per code: logical error rate vs distance."""
    for i, (code, sub) in enumerate(df.groupby("code", sort=True)):
        sub = sub.sort_values("distance")
        # Log y-axis: drop non-positive LER (0 errors → not enough shots to resolve).
        dropped = sub[sub["logical_error_rate"] <= 0]
        sub = sub[sub["logical_error_rate"] > 0]
        if not dropped.empty:
            ds = ", ".join(f"d={int(d)}" for d in dropped["distance"])
            print(f"  note: {code} has LER=0 at {ds} (below resolution) — omitted")
        if sub.empty:
            continue
        ax.plot(
            sub["distance"], sub["logical_error_rate"],
            marker=CODE_MARKERS.get(code, "o"),
            color=CODES.get(code, f"C{i % 10}"),
            lw=2, ms=8, markeredgecolor="k", markeredgewidth=0.4,
            label=CODE_LABELS.get(code, code),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Code distance $d$", fontsize=13)
    ax.set_ylabel("Logical error rate", fontsize=13)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(sorted(df["distance"].unique()))
    ax.legend(fontsize=10, framealpha=0.85, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    bold_ticks(ax)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="Input CSV file(s) from run_memory.py")
    ap.add_argument("--p", type=float, default=1e-3,
                    help="Physical error rate to hold fixed (default: 1e-3). "
                         "Snaps to the nearest value present in the data.")
    ap.add_argument("--output", default=None,
                    help="Output path (extension sets format: .pdf, .svg, .png)")
    ap.add_argument("--codes", nargs="*", default=None,
                    help="Filter to specific code name(s)")
    ap.add_argument("--distances", nargs="*", type=int, default=None,
                    help="Filter to specific distance(s)")
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
    if df.empty:
        print("No data after filtering — nothing to plot.")
        return

    if args.per_logical and "k" in df.columns:
        df = df.copy()
        df["logical_error_rate"] = df["logical_error_rate"] / df["k"]

    p_sel = _closest_p(df, args.p)
    if not np.isclose(p_sel, args.p, rtol=1e-6):
        print(f"Requested p={args.p:.2e} not in data; using nearest p={p_sel:.2e}")
    df = df[np.isclose(df["p"], p_sel)]
    if df.empty:
        print("No data at the selected p — nothing to plot.")
        return

    title = args.title or f"Memory: LER vs distance @ $p$={p_sel:.0e}"
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    plot_ler_vs_distance(df, ax, title=title)
    if args.per_logical:
        ax.set_ylabel("Logical error rate per logical qubit", fontsize=13)

    out = (Path(args.output) if args.output
           else Path(args.inputs[0]).parent / "memory_ler_vs_distance.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
