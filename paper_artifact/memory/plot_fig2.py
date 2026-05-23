"""
Regenerate fig2_bb_codes.png — BB code LER vs PER.

Reads pre-computed CSV files (one per code, gpu_bposd decoder).
Does NOT re-run experiments.

Usage:
    venv/bin/python -m eval.memory_benchmark.plot_fig2
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from lightstim.plot.styles import apply_paper_style, bold_ticks, CODES as _CODE_COLORS, CODE_LABELS

RESULTS     = Path(__file__).resolve().parent / "results"
PRECOMPUTED = Path(__file__).resolve().parent / "precomputed"
OUTPUT      = RESULTS / "fig2_bb_codes.png"


def _data_dirs():
    """Return [results/, precomputed/] in priority order."""
    return [RESULTS, PRECOMPUTED]

FIG2_CODES  = ["bb_72_12_6", "bb_108_8_10", "bb_144_12_12"]
BB_ROUNDS   = {"bb_72_12_6": 6, "bb_108_8_10": 10, "bb_144_12_12": 12}
BB_COLORS   = {k: _CODE_COLORS[k] for k in FIG2_CODES}

# Points to drop (noisy / post-threshold)
EXCLUDE = {"bb_72_12_6": {7e-4}}

# Sub-threshold fit upper bound per code
P_FIT_MAX   = {"bb_72_12_6": 1e-3, "bb_108_8_10": 2e-3, "bb_144_12_12": 2e-3}
P_EXTRAP_MIN = 1e-4

DECODER_LINESTYLES = {"gpu_bposd": "-",  "mwpf": "--"}
DECODER_MARKERS    = {"gpu_bposd": "o",  "mwpf": "X"}


def load_data():
    """Load per-code CSVs; checks results/ first, falls back to precomputed/."""
    dfs = []
    for code in FIG2_CODES:
        fname = f"fig2_bb_codes_{code}_gpu_bposd.csv"
        for d in _data_dirs():
            f = d / fname
            if f.exists():
                df = pd.read_csv(f)
                if "decoder_label" not in df.columns:
                    df["decoder_label"] = "gpu_bposd"
                dfs.append(df)
                break  # found in higher-priority dir

    # Also load mwpf data if present in the combined CSV (either dir)
    for d in _data_dirs():
        combined = d / "fig2_bb_codes.csv"
        if combined.exists():
            df_c = pd.read_csv(combined)
            if "decoder_label" in df_c.columns:
                mwpf_rows = df_c[df_c["decoder_label"] == "mwpf"]
                if not mwpf_rows.empty:
                    dfs.append(mwpf_rows)
            break

    if not dfs:
        raise FileNotFoundError(
            "No fig2 CSV data found in results/ or precomputed/.\n"
            "Run: venv/bin/python paper_artifact/memory/run_all.py --figure 2"
        )
    return pd.concat(dfs, ignore_index=True)


BB_PLAIN_LABELS = {
    "bb_72_12_6":   "[[72, 12, 6]]",
    "bb_108_8_10":  "[[108, 8, 10]]",
    "bb_144_12_12": "[[144, 12, 12]]",
}


def plot(df, save_path):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.5, 5.5))

    for code_name in FIG2_CODES:
        df_code = df[df["code"] == code_name].copy()
        color = BB_COLORS.get(code_name, "gray")
        label_base = BB_PLAIN_LABELS.get(code_name, code_name)
        rounds = BB_ROUNDS.get(code_name, 1)
        excluded_p = EXCLUDE.get(code_name, set())

        for decoder_label in ["gpu_bposd", "mwpf"]:
            df_d = df_code[df_code["decoder_label"] == decoder_label].copy()
            if df_d.empty:
                continue
            df_d = df_d[~df_d["p"].apply(
                lambda x: any(abs(x - ep) < 1e-12 for ep in excluded_p)
            )]
            df_d = df_d.sort_values("p")

            ls     = DECODER_LINESTYLES[decoder_label]
            marker = DECODER_MARKERS[decoder_label]
            p_vals  = df_d["p"].values
            ler_vals = df_d["logical_error_rate"].values / rounds

            ax.loglog(p_vals, ler_vals,
                      marker=marker, color=color, linestyle=ls,
                      markeredgecolor="none",
                      label=label_base)

            # Power-law extrapolation to the left
            p_fit_max = P_FIT_MAX.get(code_name, 2e-3)
            mask = (ler_vals < 0.5) & (p_vals <= p_fit_max)
            p_min_data = p_vals[mask].min() if mask.sum() >= 2 else None
            if p_min_data is not None and p_min_data > P_EXTRAP_MIN:
                log_p = np.log10(p_vals[mask])
                log_l = np.log10(ler_vals[mask])
                slope, intercept = np.polyfit(log_p, log_l, 1)
                p_extrap = np.logspace(np.log10(P_EXTRAP_MIN), np.log10(p_min_data), 100)
                ler_extrap = 10 ** (intercept + slope * np.log10(p_extrap))
                ax.loglog(p_extrap, ler_extrap, color=color, linestyle="--",
                          linewidth=1.6, alpha=0.7, zorder=1)

    ax.set_xlabel("Physical Error Rate ($p$)")
    ax.set_ylabel("LER per Round")
    ax.set_title("BB Codes: LER vs PER", pad=10)
    ax.set_xlim(left=1e-4)
    ax.set_ylim(bottom=1e-13)

    # Deduplicate legend (BB code labels appear twice for two decoders)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = h
    ax.legend(seen.values(), seen.keys(),
              ncol=1, loc="lower right", frameon=True)

    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)
    bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)



if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} rows from {len(df['code'].unique())} codes, "
          f"{len(df['decoder_label'].unique())} decoders")
    plot(df, OUTPUT)
