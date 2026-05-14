"""
LightStim vs Stim Built-in: Rotated Surface Code LER Comparison with error bars.

Computes Wilson score confidence intervals at 95% for each data point,
then reports per-(d,p) LER ratio and overlap statistics.

Usage:
    venv/bin/python eval/memory_benchmark/compare_stim_lightstim.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from lightstim.plot.styles import apply_paper_style, PALETTE_DISTANCE, bold_ticks

apply_paper_style()

STIM_CSV = Path("eval/memory_benchmark/results/stim_rotated_sc.csv")
LS_CSV   = Path("eval/memory_benchmark/results/fig1_surface_codes.csv")
OUT_PNG  = Path("eval/memory_benchmark/results/stim_comparison_v2.png")
DISTS    = [3, 5, 7]


def wilson_ci(errors, shots, z=1.96):
    """Wilson score 95% CI for a proportion."""
    n, k = shots, errors
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2*n)) / denom
    half   = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    return max(0, centre - half), centre + half


def load_ls(csv_path):
    """Load LightStim rotated SC Z-basis data. Returns dict (d,p) -> (ler, lo, hi)."""
    df = pd.read_csv(csv_path)
    df = df[(df["code"] == "rotated_sc") & (df["basis"] == "Z")]
    out = {}
    for _, row in df.iterrows():
        lo, hi = wilson_ci(int(row["errors"]), int(row["shots"]))
        out[(int(row["distance"]), float(row["p"]))] = (float(row["logical_error_rate"]), lo, hi)
    return out


def load_stim(csv_path):
    """Load Stim benchmark data. Returns dict (d,p) -> (ler, lo, hi)."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        lo, hi = wilson_ci(int(row["errors"]), int(row["shots"]))
        out[(int(row["d"]), float(row["p"]))] = (float(row["logical_error_rate"]), lo, hi)
    return out


def print_comparison(ls_data, stim_data):
    p_vals = sorted(set(p for d, p in ls_data) & set(p for d, p in stim_data))
    print("LightStim vs Stim Built-in — Rotated Surface Code Z-Basis Memory")
    print("Noise model: after_clifford_dep=p, before_meas=p, after_reset=p, before_round_data_dep=p")
    print("CI: 95% Wilson score interval")
    print()
    print(f"  {'d':>3}  {'p':>8}  {'LS LER':>12}  {'LS 95%CI':>22}  "
          f"{'Stim LER':>12}  {'Stim 95%CI':>22}  {'Ratio':>8}  {'Overlap':>8}")
    print("-" * 108)

    all_ratios = []
    for d in DISTS:
        for p in p_vals:
            ls  = ls_data.get((d, p))
            st  = stim_data.get((d, p))
            if ls is None or st is None:
                continue
            ls_ler,  ls_lo,  ls_hi  = ls
            st_ler,  st_lo,  st_hi  = st
            ratio   = ls_ler / st_ler if st_ler > 0 else float("nan")
            # CIs overlap?
            overlap = ls_lo <= st_hi and st_lo <= ls_hi
            all_ratios.append(ratio)
            print(f"  {d:>3}  {p:>8.1e}  {ls_ler:>12.3e}  "
                  f"[{ls_lo:.3e}, {ls_hi:.3e}]  "
                  f"{st_ler:>12.3e}  [{st_lo:.3e}, {st_hi:.3e}]  "
                  f"{ratio:>8.3f}  {'YES' if overlap else 'NO':>8}")
        print()

    if all_ratios:
        print(f"Ratio summary: median={np.median(all_ratios):.3f}  "
              f"mean={np.mean(all_ratios):.3f}  "
              f"range=[{min(all_ratios):.3f}, {max(all_ratios):.3f}]")


def plot_comparison(ls_data, stim_data):
    MARKERS = {3: "o", 5: "s", 7: "^"}
    LW, MS   = 1.8, 5
    FS_T, FS_L, FS_TK, FS_LEG = 10, 10, 9, 9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

    for d in DISTS:
        color  = PALETTE_DISTANCE[d]
        marker = MARKERS[d]

        ls_pts  = sorted((p, v) for (dd, p), v in ls_data.items()  if dd == d)
        st_pts  = sorted((p, v) for (dd, p), v in stim_data.items() if dd == d)
        st_dict = {p: v for p, v in st_pts}

        if not ls_pts or not st_pts:
            continue

        # Left panel: LER vs p with error bars
        for pts, ls_style in [(ls_pts, "-"), (st_pts, "--")]:
            ps   = [x[0] for x in pts]
            lers = [x[1][0] for x in pts]
            los  = [x[1][1] for x in pts]
            his  = [x[1][2] for x in pts]
            yerr = np.array([[l - lo, hi - l]
                             for l, lo, hi in zip(lers, los, his)]).T
            ax1.errorbar(ps, lers, yerr=yerr,
                         color=color, marker=marker,
                         lw=LW if ls_style == "-" else LW*0.8,
                         ms=MS, markeredgecolor="none",
                         linestyle=ls_style,
                         alpha=1.0 if ls_style == "-" else 0.75,
                         capsize=2, capthick=0.8, elinewidth=0.8)

        # Right panel: ratio with propagated error bars
        shared_ps = sorted(set(p for p, _ in ls_pts) & set(p for p, _ in st_pts))
        ls_dict   = {p: v for p, v in ls_pts}
        ratios, ratio_los, ratio_his = [], [], []
        for p in shared_ps:
            ls_ler, ls_lo, ls_hi = ls_dict[p]
            st_ler, st_lo, st_hi = st_dict[p]
            if st_ler == 0:
                continue
            r     = ls_ler / st_ler
            # Propagate CI bounds: r_lo = ls_lo/st_hi, r_hi = ls_hi/st_lo
            r_lo  = ls_lo / st_hi if st_hi > 0 else 0
            r_hi  = ls_hi / st_lo if st_lo > 0 else r * 3
            ratios.append(r);  ratio_los.append(r_lo);  ratio_his.append(r_hi)

        if ratios:
            yerr = np.array([[r - lo, hi - r]
                             for r, lo, hi in zip(ratios, ratio_los, ratio_his)]).T
            ax2.errorbar(shared_ps, ratios, yerr=yerr,
                         color=color, marker=marker,
                         lw=LW, ms=MS, markeredgecolor="none",
                         linestyle="-", capsize=2, capthick=0.8, elinewidth=0.8,
                         label=f"$d={d}$")

    # Left panel style
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("$p$", fontsize=FS_L)
    ax1.set_ylabel("LER", fontsize=FS_L)
    ax1.set_title("LER vs $p$", fontsize=FS_T)
    ax1.tick_params(labelsize=FS_TK)
    ax1.grid(True, which="major", ls="--", alpha=0.4)
    bold_ticks(ax1)

    proxy_ls   = Line2D([], [], color="k", ls="-",  lw=LW,       label="LightStim")
    proxy_stim = Line2D([], [], color="k", ls="--", lw=LW*0.8, alpha=0.75, label="Stim Ref.")
    dist_handles = [Line2D([], [], color=PALETTE_DISTANCE[d], marker=MARKERS[d],
                           lw=LW, ms=MS, markeredgecolor="none", label=f"$d={d}$")
                    for d in DISTS]
    ax1.legend(handles=[proxy_ls, proxy_stim] + dist_handles,
               fontsize=FS_LEG, frameon=True, loc="upper left")

    # Right panel style
    ax2.axhline(1.0, color="k", ls=":", lw=1.0, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("$p$", fontsize=FS_L)
    ax2.set_ylabel(r"LER$_\mathrm{LightStim}$ / LER$_\mathrm{Stim}$", fontsize=FS_L)
    ax2.set_title("LER Ratio (95% CI)", fontsize=FS_T)
    ax2.tick_params(labelsize=FS_TK)
    ax2.grid(True, which="major", ls="--", alpha=0.4)
    bold_ticks(ax2)
    ax2.legend(fontsize=FS_LEG, frameon=True, loc="upper right")

    fig.suptitle("Rotated Surface Code — LightStim vs Stim (matched noise model)",
                 fontweight="bold", fontsize=FS_T + 1)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    if not STIM_CSV.exists():
        print(f"ERROR: {STIM_CSV} not found. Run benchmark_stim_rotated.py first.")
        exit(1)
    if not LS_CSV.exists():
        print(f"ERROR: {LS_CSV} not found.")
        exit(1)

    ls_data   = load_ls(LS_CSV)
    stim_data = load_stim(STIM_CSV)

    print_comparison(ls_data, stim_data)
    plot_comparison(ls_data, stim_data)
