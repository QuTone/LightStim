"""
Plot state injection benchmark results.

Dual y-axis: left=LER (log), right=PS Rate (linear 0-1).
Reads results/state_injection_results.csv produced by run_state_injection.py.

Modes
-----
  --mode states     One subplot per distance; curves colored by state (Z/X/Y).
                    Filters to a single protocol and post_select_mode.
  --mode modes      One subplot per distance; curves colored by PS mode.
                    Filters to a single state and protocol.

Usage
-----
    # LER+PS by state (default: full_postselection, corner):
    PYTHONPATH=. venv/bin/python benchmarks/state_injection/plot_state_injection.py

    # LER+PS by mode for Z state, middle protocol:
    PYTHONPATH=. venv/bin/python benchmarks/state_injection/plot_state_injection.py \\
        --mode modes --inject-state Z --inject-protocol middle

    # Custom input/output:
    PYTHONPATH=. venv/bin/python benchmarks/state_injection/plot_state_injection.py \\
        --input path/to/results.csv --output path/to/plot.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE, CODES

_DEFAULT_INPUT  = SCRIPT_DIR / "results" / "state_injection_results.csv"
_DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "state_injection_plot.png"

# Z=BLUE, X=RUST(red), Y=TEAL(green)
STATE_COLORS  = {"Z": CODES["rotated_sc"], "X": PALETTE[0], "Y": PALETTE[1]}
STATE_MARKERS = {"Z": "o", "X": "s", "Y": "^"}

# full_postselection=RUST, full_qec=TEAL, hybrid=VIOLET
MODE_COLORS  = {"full_postselection": PALETTE[0], "full_qec": PALETTE[1], "hybrid": PALETTE[2]}
MODE_MARKERS = {"full_postselection": "o", "full_qec": "^", "hybrid": "s"}
MODE_LABELS  = {"full_postselection": "Full PS", "full_qec": "Full QEC", "hybrid": "Hybrid"}


def _plot_panel_states(ax1, df, distance, protocol, ps_mode):
    """One panel: LER+PS curves per state, single protocol+mode."""
    sub = df[
        (df["d"] == distance) &
        (df["injection_protocol"] == protocol) &
        (df["post_select_mode"] == ps_mode)
    ]
    ax2 = ax1.twinx()
    plotted = []
    for state in ["Z", "X", "Y"]:
        sdf = sub[sub["inject_state"] == state].sort_values("p")
        if sdf.empty:
            continue
        color, marker = STATE_COLORS[state], STATE_MARKERS[state]
        ax1.loglog(sdf["p"], sdf["logical_error_rate"],
                   color=color, marker=marker, ls="-", markeredgecolor="none")
        ax2.semilogx(sdf["p"], sdf["post_selection_rate"],
                     color=color, marker=marker, ls="--", markeredgecolor="none", alpha=0.7)
        plotted.append(state)

    handles = [
        Line2D([], [], color=STATE_COLORS[s], marker=STATE_MARKERS[s],
               ls="-", markeredgecolor="none", markersize=7, label=f"$|{s}\\rangle$")
        for s in plotted
    ] + [
        Line2D([], [], color="black", ls="-",  lw=1.5, label="LER"),
        Line2D([], [], color="black", ls="--", lw=1.5, label="PS Rate"),
    ]
    ax1.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.85)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("PS Rate")
    return ax2


def _plot_panel_modes(ax1, df, distance, state, protocol):
    """One panel: LER+PS curves per mode, single state+protocol."""
    sub = df[
        (df["d"] == distance) &
        (df["inject_state"] == state) &
        (df["injection_protocol"] == protocol)
    ]
    ax2 = ax1.twinx()
    plotted = []
    for mode in ["full_postselection", "full_qec", "hybrid"]:
        mdf = sub[sub["post_select_mode"] == mode].sort_values("p")
        if mdf.empty:
            continue
        color, marker = MODE_COLORS[mode], MODE_MARKERS[mode]
        ax1.loglog(mdf["p"], mdf["logical_error_rate"],
                   color=color, marker=marker, ls="-", markeredgecolor="none")
        ax2.semilogx(mdf["p"], mdf["post_selection_rate"],
                     color=color, marker=marker, ls="--", markeredgecolor="none", alpha=0.7)
        plotted.append(mode)

    handles = [
        Line2D([], [], color=MODE_COLORS[m], marker=MODE_MARKERS[m],
               ls="-", markeredgecolor="none", markersize=7, label=MODE_LABELS[m])
        for m in plotted
    ] + [
        Line2D([], [], color="black", ls="-",  lw=1.5, label="LER"),
        Line2D([], [], color="black", ls="--", lw=1.5, label="PS Rate"),
    ]
    ax1.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.85)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("PS Rate")
    return ax2


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",  default=str(_DEFAULT_INPUT))
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--mode", choices=["states", "modes"], default="states",
        help="Plot by state (default) or by PS mode",
    )
    ap.add_argument(
        "--inject-protocol", default="corner", choices=["corner", "middle"],
        help="Protocol to show when --mode=states (default: corner)",
    )
    ap.add_argument(
        "--ps-mode", default="full_postselection",
        choices=["full_postselection", "full_qec", "hybrid"],
        help="PS mode to show when --mode=states (default: full_postselection)",
    )
    ap.add_argument(
        "--inject-state", default="Z", choices=["Z", "X", "Y"],
        help="State to show when --mode=modes (default: Z)",
    )
    ap.add_argument(
        "--distances", nargs="+", type=int, default=None,
        help="Distances to plot (default: all in CSV)",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run run_state_injection.py first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    distances = sorted(args.distances or df["d"].unique())

    apply_paper_style()

    ncols = min(len(distances), 3)
    nrows = (len(distances) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 5.5 * nrows),
                             constrained_layout=True, squeeze=False)

    for idx, d in enumerate(distances):
        row, col = divmod(idx, ncols)
        ax1 = axes[row][col]

        if args.mode == "states":
            _plot_panel_states(ax1, df, d, args.inject_protocol, args.ps_mode)
            title = f"{args.inject_protocol.capitalize()} Injection, $d={d}$"
            subtitle = args.ps_mode.replace("_", " ").title()
        else:
            _plot_panel_modes(ax1, df, d, args.inject_state, args.inject_protocol)
            title = f"$|{args.inject_state}\\rangle$ {args.inject_protocol.capitalize()}, $d={d}$"
            subtitle = "all modes"

        ax1.set_xlabel("Physical Error Rate $p$")
        ax1.set_ylabel("LER")
        ax1.set_title(f"{title}\n({subtitle})", pad=8)
        ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
        bold_ticks(ax1)

    for idx in range(len(distances), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    out = Path(args.output) if args.output else _DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
