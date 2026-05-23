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

RESULTS     = Path(__file__).resolve().parent / "results"
PRECOMPUTED = Path(__file__).resolve().parent / "precomputed"
OUTPUT      = RESULTS / "fig1_surface_codes.png"


def _resolve(filename: str) -> Path:
    """Check results/ first; fall back to precomputed/."""
    r = RESULTS / filename
    if r.exists():
        return r
    p = PRECOMPUTED / filename
    if p.exists():
        return p
    raise FileNotFoundError(f"{filename} not found in results/ or precomputed/")


CSV = _resolve("fig1_surface_codes.csv")

PALETTE_DIST = PALETTE_DISTANCE


def plot(df):
    apply_paper_style()

    fig, ax = plt.subplots(figsize=(5.0, 5.5))

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

        code_proxy.append(mlines.Line2D([], [], color="black", ls=ls,
                                        marker=marker, markersize=7,
                                        markeredgecolor="none", linewidth=2.2,
                                        label=label))

    # Distance proxies (colored dots, no line)
    for d in [3, 5, 7]:
        color = PALETTE_DIST.get(d, "gray")
        dist_proxy.append(mlines.Line2D([], [], color=color, ls="none",
                                        marker="o", markersize=8,
                                        markeredgecolor="none",
                                        label=f"$d = {d}$"))

    ax.set_xlabel("Physical Error Rate $p$")
    ax.set_ylabel("LER per LogQ")
    ax.set_title("Surface Code Family")
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax)

    # Two-part legend — Distance box lower right, Code box above it
    leg2 = ax.legend(handles=dist_proxy, title="Distance",
                     loc="lower right", frameon=True, framealpha=0.85)
    leg2.get_title().set_fontweight("bold")
    ax.add_artist(leg2)

    fig.canvas.draw()
    bb2 = leg2.get_window_extent(fig.canvas.get_renderer())
    bb2_ax = bb2.transformed(ax.transAxes.inverted())
    gap = 0.02

    leg1 = ax.legend(handles=code_proxy, title="Code",
                     loc="lower right", frameon=True, framealpha=0.85,
                     bbox_to_anchor=(1, bb2_ax.y1 + gap),
                     bbox_transform=ax.transAxes)
    leg1.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(CSV)
    if "basis" in df.columns:
        df = df[df["basis"] == "Z"]
    plot(df)
