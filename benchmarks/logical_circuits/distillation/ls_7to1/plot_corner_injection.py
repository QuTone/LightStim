"""Plot LER vs p_inj for corner state injection LS 7-to-1 distillation."""
import sys
sys.path.insert(0, '../../../..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lightstim.plot.styles import apply_paper_style, bold_ticks

apply_paper_style()

RESULTS = Path("eval/logical_circuit_benchmark/distillation/ls_7to1")
df_all = pd.read_csv(RESULTS / "LS_distillation_corner_injection_results.csv")

# Keep only the p range we want to plot
P_PLOT = [5e-3, 1e-2, 2e-2, 5e-2]
df_all = df_all[df_all["p_injected"].isin(P_PLOT)].copy()

d3 = df_all[df_all["d"] == 3].sort_values("p_injected")
d5 = df_all[(df_all["d"] == 5) & (df_all["errors"] > 0)].sort_values("p_injected")

p_ref = np.logspace(-2.5, -1.1, 100)

COLORS = {3: "#e05c2e", 5: "#3a86cc"}
MARKERS = {3: "o", 5: "s"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

# ── Left: LER vs p_inj ───────────────────────────────────────────────────────
ax1.loglog(p_ref, 7 * p_ref**3, "k--", lw=1.5, alpha=0.6, label=r"$7\,p^3$")
ax1.loglog(p_ref, p_ref,        "k:",  lw=1.0, alpha=0.35, label=r"$p$ (PER)")

for d, df in [(3, d3), (5, d5)]:
    p_v = df["p_injected"].values
    ler = df["logical_error_rate"].values
    errs = df["errors"].values
    ax1.loglog(p_v, ler, f"{MARKERS[d]}-",
               color=COLORS[d], lw=2.0, ms=8, markeredgecolor="none",
               label=f"d={d}")
    for p, l, e in zip(p_v, ler, errs):
        ax1.annotate(f"{int(e)} err", (p, l),
                     textcoords="offset points", xytext=(5, -5),
                     fontsize=7, color="#444444")

ax1.set_xlabel(r"Injection noise $p_{\rm inj}$", fontsize=11)
ax1.set_ylabel("LER (post-selected, W4)",        fontsize=11)
ax1.set_title("LS 7→1 Distillation\nCorner State Injection", fontsize=10.5)
ax1.legend(fontsize=9, loc="upper left")
ax1.tick_params(labelsize=9)
bold_ticks(ax1)

# ── Right: acceptance rate ────────────────────────────────────────────────────
for d, df in [(3, df_all[df_all["d"]==3].sort_values("p_injected")),
              (5, df_all[df_all["d"]==5].sort_values("p_injected"))]:
    p_v   = df["p_injected"].values
    ps    = df["post_selection_rate"].values * 100
    ax2.semilogy(p_v, ps, f"{MARKERS[d]}-",
                 color=COLORS[d], lw=2.0, ms=8, markeredgecolor="none",
                 label=f"d={d}")
    for p, r in zip(p_v, ps):
        if r > 0:
            ax2.annotate(f"{r:.1f}%" if r >= 0.1 else f"{r:.2f}%",
                         (p, r), textcoords="offset points",
                         xytext=(5, 3), fontsize=7, color="#444444")

ax2.set_xlabel(r"Injection noise $p_{\rm inj}$", fontsize=11)
ax2.set_ylabel("Acceptance rate (%)",            fontsize=11)
ax2.set_title("State-Injection Post-Selection\nAcceptance Rate", fontsize=10.5)
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=9)
bold_ticks(ax2)

out = RESULTS / "fig_corner_injection.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# Print summary table
print(f"\n{'d':>3} {'p_inj':>8} {'PS%':>7} {'errors':>7} {'LER':>10} {'7p³':>10} {'ratio':>7}")
print("-" * 60)
for _, row in df_all.sort_values(["d","p_injected"]).iterrows():
    print(f"{int(row['d']):>3} {row['p_injected']:>8.0e} "
          f"{row['post_selection_rate']*100:>6.2f}% "
          f"{int(row['errors']):>7} {row['logical_error_rate']:>10.2e} "
          f"{row['p_inj_cubed_x7']:>10.2e} {row['suppression_ratio']:>7.2f}")
