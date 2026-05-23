"""
Regenerate fig3_corner_d7.png — Corner injection, d=7, Z/X/Y states.

Dual y-axis: left=LER (log), right=PS survival rate (linear 0-1).
Uses rounds=2 data from precomputed or results CSV.

Note: Y state data is missing from precomputed/. States with no data are skipped.
Run `run_all.py --inject-state Y --protocol corner` to generate Y state data.

Usage:
    PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig3.py
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
OUTPUT      = RESULTS / "fig3_corner_d7.png"

STATE_COLORS = {
    "Z": "#a63603",
    "X": "#1b9e77",
    "Y": "#7570b3",
}
STATE_MARKERS = {"Z": "o", "X": "s", "Y": "^"}

PROTOCOL = "corner"
DISTANCE = 7
ROUNDS = 2


def _load_df() -> pd.DataFrame:
    dfs = []
    results_csv = RESULTS / "state_injection.csv"
    precomputed_csv = PRECOMPUTED / "state_injection.csv"
    if results_csv.exists():
        dfs.append(pd.read_csv(results_csv))
    if precomputed_csv.exists():
        dfs.append(pd.read_csv(precomputed_csv))
    if not dfs:
        raise FileNotFoundError("state_injection.csv not found in results/ or precomputed/")
    df = pd.concat(dfs, ignore_index=True)
    key_cols = ["injection_protocol", "inject_state", "post_select_mode", "d", "rounds", "p"]
    df = df.drop_duplicates(subset=key_cols, keep="first")
    return df


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
        state_df = sub[sub["inject_state"] == state].sort_values("p")
        if state_df.empty:
            print(f"  Note: no data for inject_state={state} — skipping.")
            continue
        color = STATE_COLORS[state]
        marker = STATE_MARKERS[state]
        ax1.loglog(state_df["p"], state_df["logical_error_rate"],
                   color=color, marker=marker, ls="-", markeredgecolor="none")
        ax2.semilogx(state_df["p"], state_df["post_selection_rate"],
                     color=color, marker=marker, ls="--", markeredgecolor="none",
                     alpha=0.7)
        plotted_states.append(state)

    if not plotted_states:
        print("  No data to plot. Run run_all.py first.")
        plt.close(fig)
        return

    ax1.set_xlabel("Physical Error Rate $p$")
    ax1.set_ylabel("Logical Error Rate (LER)")
    ax2.set_ylabel("PS Survival Rate")
    ax2.set_ylim(0, 1.05)
    ax1.set_title(f"Corner Injection, $d={DISTANCE}$")
    ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax1)

    handles = []
    for state in plotted_states:
        color = STATE_COLORS[state]
        marker = STATE_MARKERS[state]
        handles.append(Line2D([], [], color=color, marker=marker, ls="-",
                               markeredgecolor="none", label=f"|{state}$\\rangle$"))
    leg = ax1.legend(handles=handles, title="State", loc="lower right",
                     frameon=True, framealpha=0.85)
    leg.get_title().set_fontweight("bold")

    ax1.annotate("solid=LER, dashed=PS rate", xy=(0.02, 0.02), xycoords="axes fraction",
                 fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    df = _load_df()
    plot(df)
