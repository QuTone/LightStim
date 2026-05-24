"""
Regenerate fig4_corner_d7_z_modes.png — Corner injection, d=7, |Z⟩ state, all 3 modes.

Dual y-axis: left=LER (log), right=PS survival rate (linear 0-1).
Color by mode. Uses rounds=2.

Usage:
    PYTHONPATH=. venv/bin/python paper_artifact/state_injection/plot_fig4.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE

RESULTS     = Path(__file__).resolve().parent / "results"
PRECOMPUTED = Path(__file__).resolve().parent / "precomputed"
OUTPUT      = RESULTS / "fig4_corner_d7_z_modes.png"

MODE_COLORS  = {
    "full_postselection": PALETTE[0],  # RUST
    "full_qec":           PALETTE[1],  # TEAL
    "hybrid":             PALETTE[2],  # VIOLET
}
MODE_MARKERS = {
    "full_postselection": "o",
    "full_qec":           "^",
    "hybrid":             "s",
}
MODE_LABELS  = {
    "full_postselection": "Full PS",
    "full_qec":           "Full QEC",
    "hybrid":             "Hybrid",
}

PROTOCOL = "corner"
DISTANCE = 7
STATE    = "Z"
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
        (df["inject_state"] == STATE) &
        (df["d"] == DISTANCE) &
        (df["rounds"] == ROUNDS)
    ].copy()

    fig, ax1 = plt.subplots(figsize=(4.5, 5.5))
    ax2 = ax1.twinx()

    plotted_modes = []
    for mode in ["full_postselection", "full_qec", "hybrid"]:
        mdf = sub[sub["post_select_mode"] == mode].sort_values("p")
        if mdf.empty:
            continue
        color  = MODE_COLORS[mode]
        marker = MODE_MARKERS[mode]
        ax1.loglog(mdf["p"], mdf["logical_error_rate"],
                   color=color, marker=marker, ls="-", markeredgecolor="none")
        ax2.semilogx(mdf["p"], mdf["post_selection_rate"],
                     color=color, marker=marker, ls="--", markeredgecolor="none", alpha=0.7)
        plotted_modes.append(mode)

    if not plotted_modes:
        print("  No data to plot.")
        plt.close(fig)
        return

    ax1.set_xlabel("Physical Error Rate $p$")
    ax1.set_ylabel("LER")
    ax2.set_ylabel("PS Rate")
    ax2.set_ylim(0, 1.05)
    ax1.set_title(f"Corner Injection, $|Z\\rangle$, $d = {DISTANCE}$")
    ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax1)

    handles = [
        Line2D([], [], color=MODE_COLORS[m], marker=MODE_MARKERS[m],
               ls="none", markeredgecolor="none", markersize=7, label=MODE_LABELS[m])
        for m in plotted_modes
    ]
    handles += [
        Line2D([], [], color="black", ls="-",  linewidth=1.5, label="LER"),
        Line2D([], [], color="black", ls="--", linewidth=1.5, label="PS Rate"),
    ]
    leg = ax1.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    df = _load_df()
    plot(df)
