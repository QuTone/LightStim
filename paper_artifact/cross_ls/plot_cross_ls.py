"""
Reproduce CrossLS (Surface–PQRM Lattice Surgery) paper figures from precomputed data.

Figures
-------
    results/fig1_ler_vs_p.png   — LER vs PER, |Z⟩ state, d=3,5,7 × 3 PQRM codes
    results/fig2_ler_vs_d.png   — LER vs d_surf, p=5e-4, all states × 3 PQRM codes

Reads
-----
    precomputed/all_sweep_data.csv
    Columns: pqrm, d_surf, state, p_2q, decoder, backend, ler, ps_rate, ...

Usage (from repo root)
----------------------
    venv/bin/python paper_artifact/cross_ls/plot_cross_ls.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, PALETTE, bold_ticks

apply_paper_style()

HERE    = Path(__file__).parent
PRECOMP = HERE / "precomputed"
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── Load & filter: use bposd GPU as primary decoder ───────────────────────────
df_all = pd.read_csv(PRECOMP / "all_sweep_data.csv")
df     = df_all[(df_all["decoder"] == "bposd") & (df_all["backend"] == "gpu")].copy()
print(f"Loaded {len(df_all)} rows total, using {len(df)} (bposd GPU)")

# ── Style constants ───────────────────────────────────────────────────────────
FS_TITLE  = 12
FS_LABEL  = 12
FS_TICK   = 11
FS_LEGEND = 9
LW        = 2.2
MS        = 8

PQRM_CODES  = ["1-2-4", "1-3-5", "1-4-6"]
PQRM_LS     = {"1-2-4": "-",  "1-3-5": "--", "1-4-6": ":"}
PQRM_MARKER = {"1-2-4": "o",  "1-3-5": "s",  "1-4-6": "^"}
PQRM_LABEL  = {
    "1-2-4": "PQRM(1,2,4)  [[15,1,3]]",
    "1-3-5": "PQRM(1,3,5)  [[31,1,5]]",
    "1-4-6": "PQRM(1,4,6)  [[63,1,7]]",
}

STATE_COLOR = {
    "Z": "#2166ac",   # blue
    "X": "#b22222",   # firebrick red
    "Y": PALETTE[1],  # teal / green
}
STATE_LABEL = {
    "Z": r"$|Z\rangle$",
    "X": r"$|X\rangle$",
    "Y": r"$|Y\rangle$",
}


# ── Fig 1: LER vs PER, |Z⟩ state ─────────────────────────────────────────────
def plot_fig1():
    sub = df[(df["state"] == "Z") & (df["d_surf"].isin([3, 5, 7]))].copy()

    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)

    for pqrm in PQRM_CODES:
        for d in [3, 5, 7]:
            rows = sub[(sub["pqrm"] == pqrm) & (sub["d_surf"] == d)].sort_values("p_2q")
            if rows.empty:
                continue
            ax.loglog(rows["p_2q"], rows["ler"],
                      color=PALETTE_DISTANCE[d],
                      linestyle=PQRM_LS[pqrm],
                      marker=PQRM_MARKER[pqrm],
                      lw=LW, ms=MS, markeredgecolor="none")

    xlim = ax.get_xlim()
    p_ref = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
    ax.loglog(p_ref, p_ref, color="crimson", ls="--", lw=2.2, zorder=0,
              label="LER $= p$")

    ax.set_xlabel(r"Physical Error Rate $p$", fontsize=FS_LABEL)
    ax.set_ylabel("LER", fontsize=FS_LABEL)
    ax.set_title(r"CrossLS — $|Z\rangle$ State", fontsize=FS_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    bold_ticks(ax)

    dist_proxy = [
        Line2D([], [], color=PALETTE_DISTANCE[d], ls="-", lw=LW,
               marker="o", ms=MS, markeredgecolor="none", label=f"$d={d}$")
        for d in [3, 5, 7]
    ]
    pqrm_proxy = [
        Line2D([], [], color="black", ls=PQRM_LS[p], lw=LW,
               marker=PQRM_MARKER[p], ms=MS, markeredgecolor="none",
               label=PQRM_LABEL[p])
        for p in PQRM_CODES
    ]
    per_proxy = [Line2D([], [], color="crimson", ls="--", lw=2.2, label="LER $= p$")]
    ax.legend(handles=dist_proxy + pqrm_proxy + per_proxy,
              fontsize=FS_LEGEND, loc="lower right", frameon=True, framealpha=0.6)

    out = OUT_DIR / "fig1_ler_vs_p.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Fig 2: LER vs d_surf, p = 5e-4 ───────────────────────────────────────────
def plot_fig2():
    sub = df[abs(df["p_2q"] - 5e-4) < 1e-10].copy()

    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)

    for pqrm in PQRM_CODES:
        for state in ["Z", "X", "Y"]:
            rows = sub[(sub["pqrm"] == pqrm) & (sub["state"] == state)].sort_values("d_surf")
            if rows.empty:
                continue
            ax.semilogy(rows["d_surf"], rows["ler"],
                        color=STATE_COLOR[state],
                        linestyle=PQRM_LS[pqrm],
                        marker=PQRM_MARKER[pqrm],
                        lw=LW, ms=MS, markeredgecolor="none")

    ax.axhline(5e-4, color=PALETTE[3], ls=":", lw=1.5, alpha=0.9)

    ax.set_xlabel(r"Surface Code Distance $d$", fontsize=FS_LABEL)
    ax.set_ylabel("LER", fontsize=FS_LABEL)
    ax.set_title(r"CrossLS — LER vs $d$, $p=5\times10^{-4}$",
                 fontsize=FS_TITLE, fontweight="bold")
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    bold_ticks(ax)

    state_proxy = [
        Line2D([], [], color=STATE_COLOR[s], ls="-", lw=LW,
               marker="o", ms=MS, markeredgecolor="none", label=STATE_LABEL[s])
        for s in ["Z", "X", "Y"]
    ]
    pqrm_proxy = [
        Line2D([], [], color="black", ls=PQRM_LS[p], lw=LW,
               marker=PQRM_MARKER[p], ms=MS, markeredgecolor="none",
               label=PQRM_LABEL[p])
        for p in PQRM_CODES
    ]
    ref_proxy = [Line2D([], [], color=PALETTE[3], ls=":", lw=1.5,
                        label=r"$p=5\times10^{-4}$")]
    ax.legend(handles=state_proxy + pqrm_proxy + ref_proxy,
              fontsize=FS_LEGEND, loc="lower left", frameon=True, framealpha=0.6)

    out = OUT_DIR / "fig2_ler_vs_d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    print("Done.")
