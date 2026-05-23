"""
Regenerate fig4_corner_d7_z_modes.png — Corner injection, d=7, Z state only, all 3 modes.

Dual y-axis: left=LER (log), right=PS survival rate (linear 0-1).
Color by mode: full_postselection=RUST, hybrid=TEAL, full_qec=VIOLET.
Uses rounds=2 data from precomputed or results CSV.

Usage:
    PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig4.py
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
OUTPUT      = RESULTS / "fig4_corner_d7_z_modes.png"

# Color by mode
MODE_COLORS = {
    "full_postselection": "#a63603",  # RUST
    "hybrid":             "#1b9e77",  # TEAL
    "full_qec":           "#7570b3",  # VIOLET
}
MODE_MARKERS = {
    "full_postselection": "o",
    "hybrid":             "s",
    "full_qec":           "^",
}
MODE_LABELS = {
    "full_postselection": "Full PS",
    "hybrid":             "Hybrid",
    "full_qec":           "Full QEC",
}

PROTOCOL = "corner"
DISTANCE = 7
STATE = "Z"
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
        (df["inject_state"] == STATE) &
        (df["d"] == DISTANCE) &
        (df["rounds"] == ROUNDS)
    ].copy()

    fig, ax1 = plt.subplots(figsize=(4.5, 5.5))
    ax2 = ax1.twinx()

    plotted_modes = []
    for mode in ["full_postselection", "hybrid", "full_qec"]:
        mode_df = sub[sub["post_select_mode"] == mode].sort_values("p")
        if mode_df.empty:
            print(f"  Note: no data for post_select_mode={mode} — skipping.")
            continue
        color = MODE_COLORS[mode]
        marker = MODE_MARKERS[mode]
        ax1.loglog(mode_df["p"], mode_df["logical_error_rate"],
                   color=color, marker=marker, ls="-", markeredgecolor="none")
        ax2.semilogx(mode_df["p"], mode_df["post_selection_rate"],
                     color=color, marker=marker, ls="--", markeredgecolor="none",
                     alpha=0.7)
        plotted_modes.append(mode)

    if not plotted_modes:
        print("  No data to plot. Run run_all.py first.")
        plt.close(fig)
        return

    ax1.set_xlabel("Physical Error Rate $p$")
    ax1.set_ylabel("Logical Error Rate (LER)")
    ax2.set_ylabel("PS Survival Rate")
    ax2.set_ylim(0, 1.05)
    ax1.set_title(f"Corner Injection, $|Z\\rangle$, $d={DISTANCE}$")
    ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax1)

    handles = []
    for mode in plotted_modes:
        color = MODE_COLORS[mode]
        marker = MODE_MARKERS[mode]
        label = MODE_LABELS[mode]
        handles.append(Line2D([], [], color=color, marker=marker, ls="-",
                               markeredgecolor="none", label=label))
    leg = ax1.legend(handles=handles, title="Mode", loc="lower right",
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
