#!/usr/bin/env python3
"""
State Injection Evaluation — Rotated Surface Code

Sweeps over injection protocol, inject_state, post_select_mode, distance, rounds,
and physical error rate. Outputs CSV results and publication-quality figures.

Usage:
    python eval/state_injection/run_rotated_sc.py [--quick]

    --quick: reduced sweep for fast iteration (fewer distances, p values)
"""

import sys
import os
import io
import argparse
import contextlib
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lightstim.protocols.state_injection import StateInjectionExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, ExperimentTask, DecoderConfig

# =============================================================================
# Configuration
# =============================================================================

FULL_SWEEP = {
    "injection_protocol": ["corner", "middle"],
    "inject_state": ["Z", "X", "Y"],
    "post_select_mode": ["full_postselection", "full_qec", "hybrid"],
    "distance": [3, 5, 7],
    "rounds": [2, 3],
    "p": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
}

QUICK_SWEEP = {
    "injection_protocol": ["corner"],
    "inject_state": ["Z", "Y"],
    "post_select_mode": ["full_postselection", "full_qec", "hybrid"],
    "distance": [3, 5],
    "rounds": [2],
    "p": [1e-4, 5e-4, 1e-3],
}

PIPELINE_CONFIG = {
    "max_errors": 200,
    "max_shots": 100_000_000,
    "num_workers": 32,
}

OUT_DIR = Path(__file__).resolve().parent / "results_rotated"

# =============================================================================
# Task building
# =============================================================================

def build_tasks(sweep: dict) -> list:
    """Build ExperimentTask list from sweep configuration."""
    tasks = []
    combos = list(product(
        sweep["injection_protocol"],
        sweep["inject_state"],
        sweep["post_select_mode"],
        sweep["distance"],
        sweep["rounds"],
        sweep["p"],
    ))

    valid_combos = list(combos)

    print(f"Building {len(valid_combos)} tasks...")
    for protocol, state, mode, d, r, p in valid_combos:
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            exp = StateInjectionExperiment(
                distance=d, rounds=r,
                inject_state=state,
                protocol=protocol,
                post_select_mode=mode,
                noise_params=noise,
            )
            circuit = exp.build()
        tasks.append(ExperimentTask(circuit, json_metadata={
            "injection_protocol": protocol,
            "inject_state": state,
            "post_select_mode": mode,
            "d": d,
            "rounds": r,
            "p": p,
        }))
    return tasks


# =============================================================================
# Plotting
# =============================================================================

MODE_STYLES = {
    "full_postselection": {"color": "C0", "marker": "o", "label": "Full PS"},
    "full_qec":           {"color": "C1", "marker": "s", "label": "Full QEC"},
    "hybrid":             {"color": "C2", "marker": "^", "label": "Hybrid (strip)"},
}

STATE_COLORS = {
    "Z": "C0",
    "X": "C1",
    "Y": "C2",
}


def plot_ler_vs_p_by_mode(df, out_dir: Path):
    """
    For each (protocol, inject_state, rounds), plot a row of subplots (one per distance).
    Each subplot shows LER vs p for the 3 post-selection modes.
    """
    grouped = df.groupby(["injection_protocol", "inject_state", "rounds"])

    for (protocol, state, r), group in grouped:
        distances = sorted(group["d"].unique())
        n_cols = len(distances)
        if n_cols == 0:
            continue

        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5), sharey=True)
        if n_cols == 1:
            axes = [axes]

        for ax, d in zip(axes, distances):
            sub = group[group["d"] == d]
            for mode, style in MODE_STYLES.items():
                mode_df = sub[sub["post_select_mode"] == mode].sort_values("p")
                if mode_df.empty:
                    continue
                ax.plot(mode_df["p"], mode_df["logical_error_rate"],
                        color=style["color"], marker=style["marker"],
                        label=style["label"], linewidth=1.5, markersize=5)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"d={d}", fontsize=11)
            ax.set_xlabel("Physical error rate (p)")
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("Logical error rate")
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"State Injection LER: {state} state, {protocol} protocol, r={r}",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()

        fname = f"ler_vs_p_{protocol}_{state}_r{r}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_ler_vs_p_by_state(df, out_dir: Path):
    """
    For each (protocol, rounds, post_select_mode), plot a row of subplots (one per distance).
    Each subplot shows LER vs p for Z/X/Y states.
    """
    grouped = df.groupby(["injection_protocol", "rounds", "post_select_mode"])

    for (protocol, r, mode), group in grouped:
        distances = sorted(group["d"].unique())
        n_cols = len(distances)
        if n_cols == 0:
            continue

        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5), sharey=True)
        if n_cols == 1:
            axes = [axes]

        for ax, d in zip(axes, distances):
            sub = group[group["d"] == d]
            for state, color in STATE_COLORS.items():
                state_df = sub[sub["inject_state"] == state].sort_values("p")
                if state_df.empty:
                    continue
                ax.plot(state_df["p"], state_df["logical_error_rate"],
                        color=color, marker="o",
                        label=f"|{state}⟩", linewidth=1.5, markersize=5)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"d={d}", fontsize=11)
            ax.set_xlabel("Physical error rate (p)")
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("Logical error rate")
        axes[-1].legend(loc="best", fontsize=8)
        mode_label = MODE_STYLES.get(mode, {}).get("label", mode)
        fig.suptitle(
            f"State Injection LER: {protocol}, r={r}, {mode_label}",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()

        fname = f"ler_vs_p_states_{protocol}_r{r}_{mode}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_ps_rate_vs_p(df, out_dir: Path):
    """
    Post-selection rate vs p for each (protocol, rounds).
    One subplot per distance, lines for each mode.
    """
    grouped = df.groupby(["injection_protocol", "inject_state", "rounds"])

    for (protocol, state, r), group in grouped:
        distances = sorted(group["d"].unique())
        n_cols = len(distances)
        if n_cols == 0:
            continue

        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5), sharey=True)
        if n_cols == 1:
            axes = [axes]

        for ax, d in zip(axes, distances):
            sub = group[group["d"] == d]
            for mode, style in MODE_STYLES.items():
                mode_df = sub[sub["post_select_mode"] == mode].sort_values("p")
                if mode_df.empty:
                    continue
                ax.plot(mode_df["p"], mode_df["post_selection_rate"],
                        color=style["color"], marker=style["marker"],
                        label=style["label"], linewidth=1.5, markersize=5)

            ax.set_xscale("log")
            ax.set_title(f"d={d}", fontsize=11)
            ax.set_xlabel("Physical error rate (p)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("Post-selection survival rate")
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"Post-selection Rate: {state} state, {protocol}, r={r}",
            fontsize=12, y=1.02,
        )
        fig.tight_layout()

        fname = f"ps_rate_{protocol}_{state}_r{r}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_summary_grid(df, out_dir: Path):
    """
    Summary figure: 2x3 grid (rows=rounds, cols=distance) for corner Z injection.
    Each panel shows LER vs p for 3 modes. Compact overview for paper.
    """
    sub = df[(df["injection_protocol"] == "corner") & (df["inject_state"] == "Z")]
    if sub.empty:
        return

    rounds_list = sorted(sub["rounds"].unique())
    distances = sorted(sub["d"].unique())
    n_rows, n_cols = len(rounds_list), len(distances)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              sharex=True, sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(rounds_list):
        for j, d in enumerate(distances):
            ax = axes[i, j]
            panel = sub[(sub["rounds"] == r) & (sub["d"] == d)]
            for mode, style in MODE_STYLES.items():
                mode_df = panel[panel["post_select_mode"] == mode].sort_values("p")
                if mode_df.empty:
                    continue
                ax.plot(mode_df["p"], mode_df["logical_error_rate"],
                        color=style["color"], marker=style["marker"],
                        label=style["label"], linewidth=1.5, markersize=4)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)

            if i == 0:
                ax.set_title(f"d={d}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"r={r}\nLogical error rate", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("p")

    axes[0, -1].legend(loc="best", fontsize=7)
    fig.suptitle("Z State Injection (corner): LER vs Physical Error Rate",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    fname = "summary_corner_Z.png"
    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

CHECKPOINT_KEYS = ["injection_protocol", "inject_state", "post_select_mode", "d", "rounds", "p"]
CSV_PATH = OUT_DIR / "state_injection_eval.csv"


def load_completed_keys() -> set:
    if not CSV_PATH.exists():
        return set()
    df = pd.read_csv(CSV_PATH)
    if any(k not in df.columns for k in CHECKPOINT_KEYS):
        return set()
    return set(zip(*(df[k] for k in CHECKPOINT_KEYS)))


def filter_tasks(tasks: list, completed: set) -> list:
    remaining, skipped = [], 0
    for t in tasks:
        m = t.json_metadata
        key = tuple(m[k] for k in CHECKPOINT_KEYS)
        if key in completed:
            skipped += 1
        else:
            remaining.append(t)
    if skipped:
        print(f"Skipping {skipped} already-completed tasks (checkpoint).")
    return remaining


def main():
    parser = argparse.ArgumentParser(description="State Injection Evaluation")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for fast iteration")
    args = parser.parse_args()

    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("State Injection Evaluation")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    # 1. Build tasks with checkpoint filtering
    all_tasks = build_tasks(sweep)
    completed = load_completed_keys()
    tasks = filter_tasks(all_tasks, completed)
    print(f"Tasks: {len(tasks)} to run / {len(all_tasks)} total")

    # 2. Run simulation
    if not tasks:
        print("All tasks already done.")
    else:
        print("\nRunning simulation...")
        pipeline = SimulationPipeline(
            decoder_config=DecoderConfig("pymatching", backend="cpu"),
            print_progress=True,
            **PIPELINE_CONFIG,
        )
        for j, task in enumerate(tasks):
            meta = task.json_metadata
            print(f"  [{j+1}/{len(tasks)}] {meta}", flush=True)
            stats = pipeline.run(task.circuit, meta)
            row = {
                **meta,
                "shots": stats.shots,
                "post_selected_shots": stats.post_selected_shots,
                "post_selection_rate": stats.post_selection_rate,
                "errors": stats.errors,
                "logical_error_rate": stats.logical_error_rate,
                "seconds": stats.seconds,
                "decoder": stats.decoder,
            }
            pd.DataFrame([row]).to_csv(
                CSV_PATH, mode="a", header=not CSV_PATH.exists(), index=False)
            print(f"    → LER={stats.logical_error_rate:.2e} "
                  f"({stats.errors}/{stats.shots:,})", flush=True)
        print(f"\nSaved CSV: {CSV_PATH}")

    # 3. Load full dataset for plots/pivot
    df = pd.read_csv(CSV_PATH)
    sort_cols = [c for c in CHECKPOINT_KEYS if c in df.columns]
    df = df.sort_values(sort_cols, ignore_index=True)

    ler_wide = df.pivot_table(
        index=["injection_protocol", "inject_state", "rounds", "d", "p"],
        columns="post_select_mode",
        values="logical_error_rate",
    ).reset_index()
    ler_wide.to_csv(OUT_DIR / "state_injection_ler_wide.csv", index=False)
    print(f"Saved wide CSV: {OUT_DIR / 'state_injection_ler_wide.csv'}")

    # 4. Generate plots
    print("\nGenerating plots...")
    plot_ler_vs_p_by_mode(df, OUT_DIR)
    plot_ler_vs_p_by_state(df, OUT_DIR)
    plot_ps_rate_vs_p(df, OUT_DIR)
    plot_summary_grid(df, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
