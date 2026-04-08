"""
Regenerate fig3_code_comparison.png — matches sample style.

Style reference: fig3_code_comparison_sample.png
  - Linear x-axis, log y-axis
  - Surface/Color codes: connected lines, d= labels on each point
  - BB codes + 4D code: individual star markers, no connecting line
  - Two legend boxes: "Surface Codes" (upper right), "BB Codes" (lower right)

Y-axis: LER / k / rounds  (per logical qubit per round)
X-axis: n_total / k       (physical qubits per logical qubit)
Fixed p = 1e-3

Usage:
    venv/bin/python -m eval.memory_benchmark.plot_fig3
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

from src.plot.styles import apply_paper_style, bold_ticks

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT  = RESULTS / "fig3_code_comparison.png"

BB_ROUNDS = {"bb_72_12_6": 6, "bb_108_8_10": 10, "bb_144_12_12": 12}

# ── Surface/Color code style ──────────────────────────────────────────
SC_STYLE = {
    "rotated_sc":   dict(color="#1f77b4", ls="-",   marker="o", label="Rotated SC"),
    "unrotated_sc": dict(color="#2ca02c", ls="--",  marker="s", label="Unrotated SC"),
    "toric":        dict(color="#e377c2", ls=":",   marker="^", label="Toric"),
    "color_code":   dict(color="#ff7f0e", ls="-.",  marker="D", label="Color (6-6-6)"),
}

# ── BB / 4D code style ────────────────────────────────────────────────
BB_STYLE = {
    "bb_72_12_6":      dict(color="#8B3A3A", marker="*", label="[[72, 12, 6]]"),
    "bb_108_8_10":     dict(color="#2aa198", marker="*", label="[[108, 8, 10]]"),
    "bb_144_12_12":    dict(color="#6c71c4", marker="*", label="[[144, 12, 12]]"),
    "4d_geo_hadamard": dict(color="#d62728", marker="*", label="[[96, 6, 8]] 4D Geo"),
}


# ── Load Data ─────────────────────────────────────────────────────────

def load_all():
    dfs = []

    # Surface codes (fig1)
    f = RESULTS / "fig1_surface_codes.csv"
    if f.exists():
        df = pd.read_csv(f)
        df = df[np.isclose(df["p"], 1e-3)].copy()
        if "rounds" not in df.columns:
            df["rounds"] = df["distance"]
        dfs.append(df)

    # BB codes — gpu_bposd only, p=1e-3
    for f in sorted(RESULTS.glob("fig2_bb_codes_*_gpu_bposd.csv")):
        df = pd.read_csv(f)
        df = df[np.isclose(df["p"], 1e-3)].copy()
        if df.empty:
            continue
        code_name = df["code"].iloc[0]
        if "rounds" not in df.columns:
            df["rounds"] = BB_ROUNDS.get(code_name, 1)
        if "distance" not in df.columns:
            df["distance"] = np.nan
        dfs.append(df)

    # Color code (fig3)
    f = RESULTS / "fig3_color_code.csv"
    if f.exists():
        df = pd.read_csv(f)
        df = df[np.isclose(df["p"], 1e-3)].copy()
        dfs.append(df)

    # 4D Hadamard
    f = RESULTS / "fig3_4d_hadamard.csv"
    if f.exists():
        df = pd.read_csv(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ── Plot ──────────────────────────────────────────────────────────────

def plot(df):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    sc_legend_handles = []
    bb_legend_handles = []

    # ── Surface / Color code lines ────────────────────────────────────
    for code_name, style in SC_STYLE.items():
        sub = df[df["code"] == code_name].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("n_total")

        x = (sub["n_total"] / sub["k"]).values
        y = (sub["logical_error_rate"] / sub["k"] / sub["rounds"]).values
        d_labels = sub["distance"].values if "distance" in sub.columns else [None]*len(sub)

        ax.semilogy(x, y,
                    color=style["color"], linestyle=style["ls"],
                    marker=style["marker"], markersize=7,
                    markeredgecolor="none", linewidth=1.8,
                    zorder=3)

        # Label each point with d=
        for xi, yi, di in zip(x, y, d_labels):
            if di is not None and not np.isnan(float(di)):
                ax.annotate(f"d={int(di)}", xy=(xi, yi),
                            xytext=(5, 3), textcoords="offset points",
                            fontsize=8, color=style["color"])

        handle = mlines.Line2D([], [],
                               color=style["color"], linestyle=style["ls"],
                               marker=style["marker"], markersize=6,
                               markeredgecolor="none", linewidth=1.8,
                               label=style["label"])
        sc_legend_handles.append(handle)

    # ── BB / 4D scatter points ────────────────────────────────────────
    for code_name, style in BB_STYLE.items():
        sub = df[df["code"] == code_name].copy()
        if sub.empty:
            continue

        x_val = (sub["n_total"] / sub["k"]).values[0]
        y_val = (sub["logical_error_rate"] / sub["k"] / sub["rounds"]).values[0]

        ax.semilogy([x_val], [y_val],
                    marker=style["marker"], color=style["color"],
                    markersize=13, markeredgecolor="none",
                    linestyle="none", zorder=4)

        handle = mlines.Line2D([], [],
                               color=style["color"],
                               marker=style["marker"], markersize=10,
                               markeredgecolor="none", linestyle="none",
                               label=style["label"])
        bb_legend_handles.append(handle)

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xlabel("Physical Qubits per LogQ ($n/k$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("LER per Round per LogQ", fontsize=12, fontweight="bold")
    ax.set_title(r"Code Comparison ($p = 10^{-3}$)", fontsize=13, fontweight="bold")
    ax.set_xlim(left=0)
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax)

    # ── Two legend boxes ──────────────────────────────────────────────
    leg1 = ax.legend(handles=sc_legend_handles, title="Surface Codes",
                     title_fontsize=9, fontsize=8.5,
                     loc="upper right", frameon=True, framealpha=0.9)
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=bb_legend_handles, title="BB Codes",
                     title_fontsize=9, fontsize=8.5,
                     loc="lower right", frameon=True, framealpha=0.9)
    leg2.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    df = load_all()
    print("Loaded data:")
    for _, row in df.sort_values("n_total").iterrows():
        rounds = row.get("rounds", 1)
        ler_norm = row["logical_error_rate"] / row["k"] / rounds
        d = row.get("distance", "—")
        print(f"  {row['code']:22s}  N/k={row['n_total']/row['k']:6.1f}  "
              f"LER/k/r={ler_norm:.3e}  d={d}")
    plot(df)
