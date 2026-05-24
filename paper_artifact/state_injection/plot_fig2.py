"""
Regenerate fig2_middle_d5.png — Middle injection, d=5, Z/X/Y states.

Dual y-axis: left=LER (log), right=PS survival rate (linear 0-1).
Uses rounds=2, full_postselection mode.

Usage:
    PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, bold_ticks

RESULTS     = Path(__file__).resolve().parent / "results"
PRECOMPUTED = Path(__file__).resolve().parent / "precomputed"
OUTPUT      = RESULTS / "fig2_middle_d5.png"

STATE_COLORS  = {"Z": "#1f77b4", "X": "#d62728", "Y": "#2ca02c"}
STATE_MARKERS = {"Z": "o",       "X": "s",        "Y": "^"}

PROTOCOL = "middle"
DISTANCE = 5
ROUNDS   = 2


def _load_df() -> pd.DataFrame:
    dfs = []
    for csv in [RESULTS / "state_injection.csv", PRECOMPUTED / "state_injection.csv"]:
        if csv.exists():
            dfs.append(pd.read_csv(csv))
    if not dfs:
        raise FileNotFoundError("state_injection.csv not found in results/ or precomputed/")
    df = pd.concat(dfs, ignore_index=True)
    key_cols = ["injection_protocol", "inject_state", "post_select_mode", "d", "rounds", "p"]
    return df.drop_duplicates(subset=key_cols, keep="first")


def plot(df):
    apply_paper_style()

    sub = df[
        (df["injection_protocol"] == PROTOCOL) &
        (df["d"] == DISTANCE) &
        (df["rounds"] == ROUNDS) &
        (df["post_select_mode"] == "full_postselection")
    ].copy()

    fig, ax1 = plt.subplots(figsize=(4.5, 5.5))
    ax2 = ax1.twinx()

    plotted_states = []
    for state in ["Z", "X", "Y"]:
        sdf = sub[sub["inject_state"] == state].sort_values("p")
        if sdf.empty:
            continue
        color  = STATE_COLORS[state]
        marker = STATE_MARKERS[state]
        ax1.loglog(sdf["p"], sdf["logical_error_rate"],
                   color=color, marker=marker, ls="-", markeredgecolor="none")
        ax2.semilogx(sdf["p"], sdf["post_selection_rate"],
                     color=color, marker=marker, ls="--", markeredgecolor="none", alpha=0.7)
        plotted_states.append(state)

    if not plotted_states:
        print("  No data to plot.")
        plt.close(fig)
        return

    ax1.set_xlabel("Physical Error Rate $p$")
    ax1.set_ylabel("LER")
    ax2.set_ylabel("PS Rate")
    ax2.set_ylim(0, 1.05)
    ax1.set_title(f"Middle Injection, $d = {DISTANCE}$")
    ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax1)

    handles = [
        Line2D([], [], color=STATE_COLORS[s], marker=STATE_MARKERS[s],
               ls="none", markeredgecolor="none", markersize=7, label=f"$|{s}\\rangle$")
        for s in plotted_states
    ]
    handles += [
        Line2D([], [], color="black", ls="-",  linewidth=1.5, label="LER"),
        Line2D([], [], color="black", ls="--", linewidth=1.5, label="PS Rate"),
    ]
    leg = ax1.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    df = _load_df()
    plot(df)
