"""
Plot logical operations benchmark results: logical error rate vs physical error rate.

Reads any CSV produced by run_logical_ops.py.
Groups curves by (gate, d) — one subplot per gate, one line per distance.

Usage
-----
    # Default input (results/logical_ops_results.csv):
    PYTHONPATH=. venv/bin/python benchmarks/logical_ops/plot_logical_ops.py

    # Custom input:
    PYTHONPATH=. venv/bin/python benchmarks/logical_ops/plot_logical_ops.py \\
        --input benchmarks/logical_ops/results/logical_ops_results.csv

    # Single gate:
    PYTHONPATH=. venv/bin/python benchmarks/logical_ops/plot_logical_ops.py \\
        --gate H CNOT_trans

    # Custom output:
    PYTHONPATH=. venv/bin/python benchmarks/logical_ops/plot_logical_ops.py \\
        --output benchmarks/logical_ops/results/my_plot.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE_DISTANCE

apply_paper_style()

_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]

_DEFAULT_INPUT  = SCRIPT_DIR / "results" / "logical_ops_results.csv"
_DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "logical_ops_plot.png"

_ROTATED_S_TITLES = {
    "S_then_S_DAG": r"Rotated S$\rightarrow$S$^\dagger$ roundtrip",
    "S_DAG_plusY_to_X": r"Rotated S$^\dagger$: $+Y\rightarrow +X$",
    "S_plusY_to_minusX": r"Rotated S: $+Y\rightarrow -X$",
}


def _plot_gate_panel(ax: plt.Axes, df_gate: pd.DataFrame, title: str) -> None:
    """
    Plot one panel (one gate): one curve per distance.

    For gates with multiple sub-experiments (e.g. CNOT, H) the LER values
    are averaged across sub-experiments before plotting.
    """
    df_avg = (
        df_gate
        .groupby(["d", "p"])["logical_error_rate"]
        .mean()
        .reset_index()
        .sort_values(["d", "p"])
    )

    distances = sorted(df_avg["d"].unique())
    for i, d in enumerate(distances):
        sub = df_avg[df_avg["d"] == d].sort_values("p")
        color  = PALETTE_DISTANCE.get(int(d), f"C{i % 10}")
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(
            sub["p"], sub["logical_error_rate"],
            marker=marker, color=color, lw=2, ms=7,
            markeredgecolor="k", markeredgewidth=0.4,
            label=f"d={d}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate $p$", fontsize=11)
    ax.set_ylabel("Logical error rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    bold_ticks(ax)


def _panel_specs(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return plot panels without averaging unlike rotated-S experiments."""
    panels = []
    for gate in sorted(df["gate"].unique()):
        df_gate = df[df["gate"] == gate]
        if gate != "S_rotated":
            panels.append((gate, df_gate))
            continue
        for sub_experiment in _ROTATED_S_TITLES:
            df_sub = df_gate[
                df_gate["sub_experiment"] == sub_experiment
            ]
            if not df_sub.empty:
                panels.append(
                    (_ROTATED_S_TITLES[sub_experiment], df_sub)
                )
    return panels


def plot_results(df: pd.DataFrame) -> plt.Figure:
    """Build the logical-operations figure for an already-filtered table."""
    if df.empty:
        raise ValueError("No data to plot.")

    panels = _panel_specs(df)
    n = len(panels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, (title, df_panel) in enumerate(panels):
        row, col = divmod(idx, ncols)
        _plot_gate_panel(axes[row][col], df_panel, title)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", default=str(_DEFAULT_INPUT),
        help=f"Input CSV file from run_logical_ops.py (default: {_DEFAULT_INPUT})",
    )
    ap.add_argument(
        "--gate", nargs="*", default=None,
        help="Filter to specific gate(s) (default: all gates in the CSV)",
    )
    ap.add_argument(
        "--distances", nargs="*", type=int, default=None,
        help="Filter to specific distance(s) (default: all)",
    )
    ap.add_argument(
        "--output", default=None,
        help=f"Output PNG path (default: {_DEFAULT_OUTPUT})",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        print("Run run_logical_ops.py first to generate results.")
        sys.exit(1)

    df = pd.read_csv(input_path)

    if args.gate:
        df = df[df["gate"].isin(args.gate)]
    if args.distances:
        df = df[df["d"].isin(args.distances)]

    if df.empty:
        print("No data after filtering — nothing to plot.")
        return

    fig = plot_results(df)

    if args.output:
        out = Path(args.output)
    else:
        out = _DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
