"""Plot LER vs PER for TG 7-to-1 distillation (full circuit-level noise).

Usage:
    venv/bin/python eval/logical_circuit_benchmark/distillation/tg_7to1/plot_tg_7to1.py
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[4]))  # project root

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.plot.styles import apply_paper_style, bold_ticks

apply_paper_style()

HERE = Path(__file__).parent
OUT  = HERE / "results"
OUT.mkdir(exist_ok=True)

# ── Load data: prefer CSV (bposd run), fall back to mwpf JSON ─────────────────
import csv
from collections import defaultdict

csv_path = HERE / "TG_distillation_7_to_1_results.csv"
json_path = HERE / "results_mwpf_v2.json"

records = []
if csv_path.exists():
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            records.append({
                "d":                    int(row["d"]),
                "p":                    float(row["p"]),
                "decoder":              row.get("decoder", "bposd"),
                "logical_error_rate":   float(row["logical_error_rate"]),
                "post_selection_rate":  float(row["post_selection_rate"]),
                "errors":               int(row["errors"]),
                "shots":                int(row["shots"]),
                "post_selected_shots":  int(row["post_selected_shots"]),
            })
    print(f"Loaded {len(records)} records from {csv_path.name}")
else:
    with open(json_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {json_path.name}")

# Group by d, sort by p
by_d = defaultdict(list)
for r in records:
    by_d[r["d"]].append(r)
for d in by_d:
    by_d[d].sort(key=lambda r: r["p"])

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS  = {3: "#a63603", 5: "#1b9e77", 7: "#7570b3"}   # RUST / TEAL / VIOLET
MARKERS = {3: "o",       5: "s",       7: "^"}
LABELS  = {3: r"$d=3$",  5: r"$d=5$",  7: r"$d=7$"}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
ax1, ax2 = axes

# ── Left: LER vs PER ─────────────────────────────────────────────────────────
for d in sorted(by_d):
    rows = by_d[d]
    p_v  = np.array([r["p"]                 for r in rows])
    ler  = np.array([r["logical_error_rate"] for r in rows])
    ax1.loglog(p_v, ler,
               f"{MARKERS[d]}-",
               color=COLORS[d], lw=2.0, ms=8, markeredgecolor="none",
               label=LABELS[d])

decoder_label = records[0].get("decoder", "bposd").upper() if records else "BPOSD"
ax1.set_xlabel(r"Physical error rate $p$",       fontsize=11)
ax1.set_ylabel("Logical error rate (post-sel.)",  fontsize=11)
ax1.set_title(f"TG 7→1 Distillation\nFull circuit-level noise ({decoder_label})", fontsize=10.5)
ax1.legend(fontsize=9, loc="upper left")
ax1.tick_params(labelsize=9)
bold_ticks(ax1)

# ── Right: post-selection acceptance rate ─────────────────────────────────────
for d in sorted(by_d):
    rows = by_d[d]
    p_v = np.array([r["p"]                   for r in rows])
    ps  = np.array([r["post_selection_rate"]  for r in rows]) * 100
    ax2.semilogx(p_v, ps,
                 f"{MARKERS[d]}-",
                 color=COLORS[d], lw=2.0, ms=8, markeredgecolor="none",
                 label=LABELS[d])

ax2.set_xlabel(r"Physical error rate $p$", fontsize=11)
ax2.set_ylabel("Acceptance rate (%)",       fontsize=11)
ax2.set_title("Steane post-selection\nacceptance rate", fontsize=10.5)
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9)
ax2.tick_params(labelsize=9)
bold_ticks(ax2)

out_path = OUT / "tg_7to1_ler_vs_p.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'d':>3} {'p':>8} {'PS%':>7} {'shots':>8} {'ps_shots':>9} "
      f"{'errors':>7} {'LER':>10} {'7p³':>10}")
print("-" * 70)
for d in sorted(by_d):
    for r in by_d[d]:
        print(f"{r['d']:>3} {r['p']:>8.1e} "
              f"{r['post_selection_rate']*100:>6.2f}% "
              f"{r['shots']:>8,} {r['post_selected_shots']:>9,} "
              f"{r['errors']:>7} {r['logical_error_rate']:>10.3e} "
              f"{7*r['p']**3:>10.3e}")
