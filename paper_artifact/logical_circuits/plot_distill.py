"""
Plot distillation P_out vs P_in for TG and LS 7-to-1 protocols.

Reads:
    precomputed/distill_tg_injection.csv
    precomputed/distill_ls_injection.csv

    Columns: d, rounds, [r,] p, p_injected, p_in, ler_ps,
             post_selection_rate, shots, errors

Outputs:
    results/distill_tg.png          — TG only
    results/distill_ls.png          — LS only
    results/distill.png             — TG + LS side-by-side (paper figure)

Usage (from repo root):
    venv/bin/python paper_artifact/logical_circuits/plot_distill.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightstim.plot.styles import apply_paper_style, bold_ticks, PALETTE_DISTANCE

apply_paper_style()

HERE    = Path(__file__).parent
PRECOMP = HERE / "precomputed"
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

MARKERS   = ["o", "s", "^", "D", "v"]
UB_THRESH = 5   # rows with fewer errors → upper bound only


def load(fname):
    path = PRECOMP / fname
    if not path.exists():
        print(f"  [skip] {fname} — not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["p_in"] > 0].copy()
    print(f"  Loaded {len(df)} valid rows ← {fname}")
    return df


def wilson_errbar(df):
    n = (df["shots"] * df["post_selection_rate"]).clip(lower=1)
    k = df["errors"].astype(float)
    p_hat = k / n
    z = 3.0
    denom  = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    lo = (centre - half).clip(lower=0)
    hi = centre + half
    return (p_hat - lo).values, (hi - p_hat).values


def plot_injection(ax, df, title="", add_legend=True):
    if df.empty:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color="gray")
        ax.set_title(title, fontsize=14, fontweight="bold")
        return

    distances = sorted(df["d"].unique())
    all_pin, all_ler = [], []

    for i, d in enumerate(distances):
        sub    = df[df["d"] == d].sort_values("p_in")
        color  = PALETTE_DISTANCE.get(d, PALETTE_DISTANCE.get(3))
        marker = MARKERS[i % len(MARKERS)]
        label  = f"$d={d}$"

        is_ub = sub["errors"] < UB_THRESH
        meas  = sub[~is_ub]
        ub    = sub[is_ub]

        if not meas.empty:
            lo, hi = wilson_errbar(meas)
            ax.errorbar(meas["p_in"], meas["ler_ps"],
                        yerr=[lo, hi],
                        fmt=marker + "-", color=color, capsize=3,
                        lw=2, ms=7, markeredgecolor="k", markeredgewidth=0.4,
                        label=label)
            all_pin.extend(meas["p_in"].tolist())
            all_ler.extend(meas["ler_ps"].tolist())

        if not ub.empty:
            n_post = (ub["shots"] * ub["post_selection_rate"]).clip(lower=1)
            ub_val = 3.0 / n_post
            first  = True
            for (_, row), uv in zip(ub.iterrows(), ub_val):
                lbl = label if (meas.empty and first) else "_nolegend_"
                ax.annotate("",
                            xy=(row["p_in"], uv * 0.20),
                            xytext=(row["p_in"], uv),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=color, lw=1.8,
                                            mutation_scale=10))
                ax.plot(row["p_in"], uv, "v", color=color, ms=7, alpha=0.7,
                        markeredgecolor="k", markeredgewidth=0.4, label=lbl)
                first = False

    if all_pin and all_ler:
        xlo = 10 ** (np.log10(min(all_pin)) - 0.5)
        xhi = 10 ** (np.log10(max(all_pin)) + 0.3)
        ylo = 10 ** (np.log10(min(all_ler)) - 1.0)
        yhi = min(1.0, 10 ** (np.log10(max(all_ler)) + 0.5))
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    # Theory curve: 7 * P_in^3
    xlo, xhi = ax.get_xlim()
    pin_th = np.logspace(np.log10(max(xlo, 1e-10)), np.log10(xhi), 300)
    ax.plot(pin_th, 7 * pin_th**3, "k--", lw=1.8,
            label=r"$7\,P_{\rm in}^3$", zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$P_{\rm in}$", fontsize=14)
    ax.set_ylabel(r"$P_{\rm out}$", fontsize=14)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    if add_legend:
        ax.legend(fontsize=11, framealpha=0.85, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    bold_ticks(ax)


print("Loading...")
tg = load("distill_tg_injection.csv")
ls = load("distill_ls_injection.csv")

# Individual figures
for df, name, title in [
    (tg, "distill_tg.png", "TG 7-to-1 — Injection-only noise"),
    (ls, "distill_ls.png", "LS 7-to-1 — Injection-only noise"),
]:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    plot_injection(ax, df, title=title)
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / name}")

# Combined side-by-side (paper Distill.png)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
plot_injection(axes[0], tg, title="TG 7-to-1", add_legend=True)
plot_injection(axes[1], ls, title="LS 7-to-1", add_legend=True)

# Shared y-axis
if not tg.empty and not ls.empty:
    all_y = []
    for ax in axes:
        for line in ax.get_lines():
            all_y.extend([y for y in line.get_ydata() if np.isfinite(y) and y > 0])
    if all_y:
        ymin = 10 ** (np.log10(min(all_y)) - 0.5)
        ymax = 10 ** (np.log10(max(all_y)) + 0.5)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

fig.savefig(OUT_DIR / "distill.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'distill.png'}")
print("Done.")
