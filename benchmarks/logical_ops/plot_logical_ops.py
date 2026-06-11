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


def _plot_gate_panel(ax: plt.Axes, df_gate: pd.DataFrame, gate: str) -> None:
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
    ax.set_title(gate, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    bold_ticks(ax)


def parse_pauli_subexperiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand 'pauli' gate sub_experiment labels (P{X|Z}_{physical|frame}_L{N})
    into dedicated columns: pauli ('X'/'Z'), mode ('physical'/'frame'),
    layers (int N).
    """
    parsed = df["sub_experiment"].str.extract(
        r"^P(?P<pauli>[XZ])_(?P<mode>physical|frame)_L(?P<layers>\d+)$"
    )
    out = df.copy()
    out["pauli"] = parsed["pauli"]
    out["mode"] = parsed["mode"]
    out["layers"] = parsed["layers"].astype(int)
    return out


def _plot_pauli_figure(df_pauli: pd.DataFrame, out_path: Path) -> None:
    """
    Dedicated §7.1 figure — physical vs frame must NOT be averaged together.

    Left:  LER vs p at the largest layer count N (solid=physical, dashed=frame).
    Right: LER vs N at a fixed p (median of swept values) — the per-layer cost
           appears as a positive slope for physical and zero slope for frame.
    LER is averaged over the PX/PZ pairings (symmetric sub-experiments).
    """
    df = parse_pauli_subexperiments(df_pauli)
    df = (
        df.groupby(["d", "p", "mode", "layers"])["logical_error_rate"]
        .mean()
        .reset_index()
    )

    fig, (ax_p, ax_n) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )
    distances = sorted(df["d"].unique())
    n_max = int(df["layers"].max())
    p_values = sorted(df["p"].unique())
    p_fixed = p_values[len(p_values) // 2]

    style = {"physical": ("-", "physical"), "frame": ("--", "frame")}

    for i, d in enumerate(distances):
        color = PALETTE_DISTANCE.get(int(d), f"C{i % 10}")
        marker = _MARKERS[i % len(_MARKERS)]
        for mode, (ls, mode_label) in style.items():
            sub = df[(df["d"] == d) & (df["mode"] == mode)
                     & (df["layers"] == n_max)].sort_values("p")
            if not sub.empty:
                ax_p.plot(
                    sub["p"], sub["logical_error_rate"],
                    ls=ls, marker=marker, color=color, lw=2, ms=6,
                    markeredgecolor="k", markeredgewidth=0.4,
                    label=f"d={d} {mode_label}",
                )
            sub = df[(df["d"] == d) & (df["mode"] == mode)
                     & (df["p"] == p_fixed)].sort_values("layers")
            if not sub.empty:
                ax_n.plot(
                    sub["layers"], sub["logical_error_rate"],
                    ls=ls, marker=marker, color=color, lw=2, ms=6,
                    markeredgecolor="k", markeredgewidth=0.4,
                    label=f"d={d} {mode_label}",
                )

    ax_p.set_xscale("log")
    ax_p.set_yscale("log")
    ax_p.set_xlabel("Physical error rate $p$", fontsize=11)
    ax_p.set_ylabel("Logical error rate", fontsize=11)
    ax_p.set_title(f"Logical Pauli, N={n_max} layers", fontsize=12, fontweight="bold")

    ax_n.set_yscale("log")
    ax_n.set_xlabel("Pauli layers $N$", fontsize=11)
    ax_n.set_ylabel("Logical error rate", fontsize=11)
    ax_n.set_title(f"Logical Pauli, p={p_fixed:g}", fontsize=12, fontweight="bold")

    for ax in (ax_p, ax_n):
        ax.legend(fontsize=8, framealpha=0.85, loc="upper left", ncols=2)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        bold_ticks(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


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

    # The 'pauli' gate gets a dedicated figure: averaging its physical/frame
    # sub-experiments together (as the generic panel does) would erase the
    # very comparison the experiment makes.
    if "pauli" in df["gate"].unique():
        pauli_out = (Path(args.output).with_name("logical_pauli_plot.png")
                     if args.output else SCRIPT_DIR / "results" / "logical_pauli_plot.png")
        _plot_pauli_figure(df[df["gate"] == "pauli"], pauli_out)
        df = df[df["gate"] != "pauli"]
        if df.empty:
            return

    gates = sorted(df["gate"].unique())
    n = len(gates)

    # Layout: up to 3 columns
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, gate in enumerate(gates):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        df_gate = df[df["gate"] == gate]
        _plot_gate_panel(ax, df_gate, gate)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

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
