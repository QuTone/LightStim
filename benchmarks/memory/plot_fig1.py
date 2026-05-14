"""
Regenerate fig1_surface_codes.png — matches committed sample style.

Legend: two-part proxy (Code style + Distance color), same as original image.

To change figure size, edit the figsize line below.

Usage:
    venv/bin/python -m eval.memory_benchmark.plot_fig1
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

from lightstim.plot.styles import (
    apply_paper_style, bold_ticks,
    PALETTE_DISTANCE, CODE_LINESTYLES, CODE_MARKERS, CODE_LABELS,
)

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT  = RESULTS / "fig1_surface_codes.png"
CSV     = RESULTS / "fig1_surface_codes.csv"

PALETTE_DIST = PALETTE_DISTANCE


def plot(df):
    apply_paper_style()

    # ── Change figsize here ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.9, 4.2))
    # ─────────────────────────────────────────────────────────────────

    code_proxy = []
    dist_proxy = []

    for code_name in ["rotated_sc", "unrotated_sc", "toric"]:
        df_code = df[df["code"] == code_name]
        ls      = CODE_LINESTYLES.get(code_name, "-")
        marker  = CODE_MARKERS.get(code_name, "o")
        label   = CODE_LABELS.get(code_name, code_name)

        for d in sorted(df_code["distance"].dropna().unique()):
            df_d  = df_code[df_code["distance"] == d].sort_values("p")
            color = PALETTE_DIST.get(int(d), "gray")
            ax.loglog(
                df_d["p"], df_d["logical_error_rate"] / df_d["k"],
                marker=marker, color=color, linestyle=ls,
                markeredgecolor="none",
            )

        # Code proxy (black, captures line style + marker)
        code_proxy.append(mlines.Line2D([], [], color="black", ls=ls,
                                        marker=marker, markersize=6,
                                        markeredgecolor="none", linewidth=1.8,
                                        label=label))

    # Distance proxies (colored dots, no line)
    for d in [3, 5, 7]:
        color = PALETTE_DIST.get(d, "gray")
        dist_proxy.append(mlines.Line2D([], [], color=color, ls="none",
                                        marker="o", markersize=7,
                                        markeredgecolor="none",
                                        label=f"d={d}"))

    ax.set_xlabel("Physical Error Rate $p$")
    ax.set_ylabel("LER per LogQ")
    ax.set_title("Surface Code Family")
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax)

    # Two-part legend — both lower right, code above distance, no titles
    # Draw distance first so we can get its height to position code above it
    leg2 = ax.legend(handles=dist_proxy, fontsize=10,
                     loc="lower right", frameon=True, framealpha=0.7)
    ax.add_artist(leg2)

    fig.canvas.draw()
    bb2 = leg2.get_window_extent(fig.canvas.get_renderer())
    bb2_ax = bb2.transformed(ax.transAxes.inverted())
    gap = 0.015  # axes fraction gap between the two boxes

    leg1 = ax.legend(handles=code_proxy, fontsize=10,
                     loc="lower right", frameon=True, framealpha=0.7,
                     bbox_to_anchor=(1, bb2_ax.y1 + gap),
                     bbox_transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    if "basis" in df.columns:
        df = df[df["basis"] == "Z"]
    plot(df)
