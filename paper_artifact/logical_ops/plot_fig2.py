"""
Regenerate fig2_cnot_ls_xx_zz.png — LS CNOT XX->ZZ logical error rate.

Aggregates LER over 5 sub-experiments, color by distance.

Usage:
    PYTHONPATH=. venv/bin/python paper_artifact/logical_ops/plot_fig2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE_DISTANCE

RESULTS     = Path(__file__).resolve().parent / "results"
PRECOMPUTED = Path(__file__).resolve().parent / "precomputed"
OUTPUT      = RESULTS / "fig2_cnot_ls_xx_zz.png"

PALETTE_DIST = PALETTE_DISTANCE
MARKERS = {3: "o", 5: "s", 7: "^"}


def _resolve(filename: str) -> Path:
    """Check results/ first; fall back to precomputed/."""
    r = RESULTS / filename
    if r.exists():
        return r
    p = PRECOMPUTED / filename
    if p.exists():
        return p
    raise FileNotFoundError(f"{filename} not found in results/ or precomputed/")


def plot(df):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.5, 5.5))

    # Aggregate LER over the 5 sub-experiments by taking mean per (d, p)
    df_agg = (
        df.groupby(["d", "p"])["logical_error_rate"]
        .mean()
        .reset_index()
    )

    dist_proxy = []
    for d in sorted(df_agg["d"].unique()):
        df_d = df_agg[df_agg["d"] == d].sort_values("p")
        color = PALETTE_DIST.get(int(d), "gray")
        marker = MARKERS.get(int(d), "o")
        ax.loglog(
            df_d["p"], df_d["logical_error_rate"],
            marker=marker, color=color, linestyle="-",
            markeredgecolor="none",
        )
        dist_proxy.append(mlines.Line2D(
            [], [], color=color, ls="-", marker=marker, markersize=8,
            markeredgecolor="none", label=f"$d = {d}$",
        ))

    ax.set_xlabel("Physical Error Rate $p$")
    ax.set_ylabel("Logical Error Rate (LER)")
    ax.set_title("LS CNOT XX→ZZ")
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax)

    leg = ax.legend(handles=dist_proxy, title="Distance",
                    loc="lower right", frameon=True, framealpha=0.85)
    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    csv = _resolve("fig2_cnot_ls_xx_zz.csv")
    df = pd.read_csv(csv)
    plot(df)
