"""
Plot LER vs. Routing distance for ZZ-LS and XX-LS protocols.

Reads:
    precomputed/routing_ls_zz_base.csv   (routing_mult=1 baseline, ZZ coupler)
    precomputed/routing_ls_zz_dist.csv   (routing_mult=2,4,8, ZZ coupler)
    precomputed/routing_ls_xx_base.csv   (routing_mult=1 baseline, XX coupler)
    precomputed/routing_ls_xx_dist.csv   (routing_mult=2,4,8, XX coupler)

Output: results/routing.png

Usage (from repo root):
    venv/bin/python paper_artifact/logical_circuits/plot_routing.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

from lightstim.plot.styles import apply_paper_style, bold_ticks

apply_paper_style()

HERE    = Path(__file__).parent
PRECOMP = HERE / "precomputed"
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

P_FIX = 1e-3
D     = 7

COLOR_Z = "#1f77b4"
COLOR_X = "#d62728"


def load_series(base_csv, dist_csv):
    base = pd.read_csv(PRECOMP / base_csv)
    dist = pd.read_csv(PRECOMP / dist_csv)
    base["routing_mult"] = 1
    df = pd.concat(
        [base[["d", "state", "routing_mult", "p", "logical_error_rate"]],
         dist[["d", "state", "routing_mult", "p", "logical_error_rate"]]],
        ignore_index=True,
    )
    df7 = df[(df["d"] == D) & np.isclose(df["p"], P_FIX)].copy()
    df7 = df7.drop_duplicates(subset=["state", "routing_mult"])

    def get(state):
        sub = df7[df7["state"] == state].sort_values("routing_mult")
        return sub["routing_mult"].values.astype(float), sub["logical_error_rate"].values

    return get("X"), get("Z")


(r_zz_X, ler_zz_X), (r_zz_Z, ler_zz_Z) = load_series(
    "routing_ls_zz_base.csv", "routing_ls_zz_dist.csv"
)
(r_xx_X, ler_xx_X), (r_xx_Z, ler_xx_Z) = load_series(
    "routing_ls_xx_base.csv", "routing_ls_xx_dist.csv"
)

for label, r, ler in [
    ("ZZ |Z>", r_zz_Z, ler_zz_Z),
    ("ZZ |X>", r_zz_X, ler_zz_X),
    ("XX |X>", r_xx_X, ler_xx_X),
    ("XX |Z>", r_xx_Z, ler_xx_Z),
]:
    print(f"{label}  r={list(r.astype(int))}  LER={[f'{v:.2e}' for v in ler]}")

fig, ax = plt.subplots(figsize=(3.0, 4.2), constrained_layout=True)

kw = dict(lw=1.8, ms=6, markeredgecolor="k", markeredgewidth=0.4)

ax.plot(r_zz_Z, ler_zz_Z * 1e3, "o-",  color=COLOR_Z, **kw,
        label=r"ZZ — Teleport $|Z\rangle$")
ax.plot(r_zz_X, ler_zz_X * 1e3, "s-",  color=COLOR_X, **kw,
        label=r"ZZ — Teleport $|X\rangle$")
ax.plot(r_xx_X, ler_xx_X * 1e3, "^--", color=COLOR_X, **kw,
        label=r"XX — Teleport $|X\rangle$")
ax.plot(r_xx_Z, ler_xx_Z * 1e3, "v--", color=COLOR_Z, **kw,
        label=r"XX — Teleport $|Z\rangle$")

# Linear fit guide lines for the two "vulnerable" series
for r_arr, ler_arr, color in [
    (r_zz_Z, ler_zz_Z, COLOR_Z),
    (r_xx_X, ler_xx_X, COLOR_X),
]:
    if len(r_arr) >= 2:
        slope, intercept, *_ = linregress(r_arr, ler_arr)
        r_fit = np.linspace(r_arr.min(), r_arr.max(), 100)
        ax.plot(r_fit, (slope * r_fit + intercept) * 1e3,
                color=color, lw=0.9, alpha=0.35, ls=":")

ax.set_xlabel("Routing multiplier $r$", fontsize=10)
ax.set_ylabel(r"LER  ($\times 10^{-3}$)", fontsize=10)
ax.set_title(
    f"LER vs Routing Distance\n" r"($d=7$, $p=10^{{-3}}$)",
    fontsize=10, fontweight="bold",
)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7.5, frameon=True, framealpha=0.7,
          loc="upper left", handlelength=2.0)
ax.grid(True, ls="--", alpha=0.4)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
bold_ticks(ax)

out = OUT_DIR / "routing.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
