#!/usr/bin/env python3
"""
State Injection Evaluation — Unrotated Surface Code

Sweeps over inject_state, post_select_mode, distance, rounds,
and physical error rate. Corner injection only.

Usage:
    python eval/state_injection/run_unrotated_sc.py [--quick]

    --quick: reduced sweep for fast iteration
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

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.state_injection import StateInjectionExperiment
from src.noise.config import NoiseConfig
from src.simulation.decoder_backend import SimulationPipeline, ExperimentTask, DecoderConfig
from src.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedSurfaceCodeLogicalOpSet,
)

# =============================================================================
# Configuration
# =============================================================================

FULL_SWEEP = {
    "inject_state": ["Z", "X", "Y"],
    "post_select_mode": ["full_postselection", "full_qec", "hybrid"],
    "distance": [3, 5, 7],
    "rounds": [2, 3],
    "p": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
}

QUICK_SWEEP = {
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

OUT_DIR = Path(__file__).resolve().parent / "results_unrotated"

# =============================================================================
# Task building
# =============================================================================

def build_tasks(sweep: dict) -> list:
    """Build ExperimentTask list from sweep configuration."""
    tasks = []
    combos = list(product(
        sweep["inject_state"],
        sweep["post_select_mode"],
        sweep["distance"],
        sweep["rounds"],
        sweep["p"],
    ))

    print(f"Building {len(combos)} tasks...")
    for state, mode, d, r, p in combos:
        noise = NoiseConfig(p_meas=p, p_reset=p, p_1q=p, p_2q=p, p_idle=p)
        with contextlib.redirect_stdout(io.StringIO()):
            exp = StateInjectionExperiment(
                code_patch_class=UnrotatedSurfaceCode,
                extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
                op_set_class=UnrotatedSurfaceCodeLogicalOpSet,
                distance=d, rounds=r,
                inject_state=state,
                protocol="corner",
                post_select_mode=mode,
                noise_params=noise,
            )
            circuit = exp.build()
        tasks.append(ExperimentTask(circuit, json_metadata={
            "code": "unrotated_sc",
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

STATE_COLORS = {"Z": "C0", "X": "C1", "Y": "C2"}


def plot_ler_vs_p_by_mode(df, out_dir: Path):
    """For each (inject_state, rounds), subplot per distance, curves per mode."""
    for (state, r), group in df.groupby(["inject_state", "rounds"]):
        distances = sorted(group["d"].unique())
        n = len(distances)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, d in zip(axes, distances):
            sub = group[group["d"] == d]
            for mode, sty in MODE_STYLES.items():
                mdf = sub[sub["post_select_mode"] == mode].sort_values("p")
                if mdf.empty:
                    continue
                ax.plot(mdf["p"], mdf["logical_error_rate"],
                        color=sty["color"], marker=sty["marker"],
                        label=sty["label"], linewidth=1.5, markersize=5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"d={d}"); ax.set_xlabel("p")
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("Logical error rate")
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(f"Unrotated SC — {state} injection, r={r}", fontsize=12, y=1.02)
        fig.tight_layout()
        fname = f"ler_vs_p_{state}_r{r}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_ler_vs_p_by_state(df, out_dir: Path):
    """For each (rounds, mode), subplot per distance, curves per state."""
    for (r, mode), group in df.groupby(["rounds", "post_select_mode"]):
        distances = sorted(group["d"].unique())
        n = len(distances)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, d in zip(axes, distances):
            sub = group[group["d"] == d]
            for state, color in STATE_COLORS.items():
                sdf = sub[sub["inject_state"] == state].sort_values("p")
                if sdf.empty:
                    continue
                ax.plot(sdf["p"], sdf["logical_error_rate"],
                        color=color, marker="o", label=f"|{state}>",
                        linewidth=1.5, markersize=5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"d={d}"); ax.set_xlabel("p")
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("Logical error rate")
        axes[-1].legend(loc="best", fontsize=8)
        mode_label = MODE_STYLES.get(mode, {}).get("label", mode)
        fig.suptitle(f"Unrotated SC — r={r}, {mode_label}", fontsize=12, y=1.02)
        fig.tight_layout()
        fname = f"ler_vs_p_states_r{r}_{mode}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_ps_rate(df, out_dir: Path):
    """Post-selection rate vs p for each (inject_state, rounds)."""
    for (state, r), group in df.groupby(["inject_state", "rounds"]):
        distances = sorted(group["d"].unique())
        n = len(distances)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, d in zip(axes, distances):
            sub = group[group["d"] == d]
            for mode, sty in MODE_STYLES.items():
                mdf = sub[sub["post_select_mode"] == mode].sort_values("p")
                if mdf.empty:
                    continue
                ax.plot(mdf["p"], mdf["post_selection_rate"],
                        color=sty["color"], marker=sty["marker"],
                        label=sty["label"], linewidth=1.5, markersize=5)
            ax.set_xscale("log")
            ax.set_title(f"d={d}"); ax.set_xlabel("p"); ax.set_ylim(0, 1.05)
            ax.grid(True, which="both", alpha=0.3)

        axes[0].set_ylabel("PS survival rate")
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(f"Unrotated SC — PS rate, {state} injection, r={r}", fontsize=12, y=1.02)
        fig.tight_layout()
        fname = f"ps_rate_{state}_r{r}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_summary_grid(df, out_dir: Path):
    """Summary: rows=rounds, cols=distance for Z injection."""
    sub = df[df["inject_state"] == "Z"]
    if sub.empty:
        return
    rounds_list = sorted(sub["rounds"].unique())
    distances = sorted(sub["d"].unique())
    nr, nc = len(rounds_list), len(distances)

    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.5 * nr), sharex=True, sharey=True)
    if nr == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(rounds_list):
        for j, d in enumerate(distances):
            ax = axes[i, j]
            panel = sub[(sub["rounds"] == r) & (sub["d"] == d)]
            for mode, sty in MODE_STYLES.items():
                mdf = panel[panel["post_select_mode"] == mode].sort_values("p")
                if mdf.empty:
                    continue
                ax.plot(mdf["p"], mdf["logical_error_rate"],
                        color=sty["color"], marker=sty["marker"],
                        label=sty["label"], linewidth=1.5, markersize=4)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if i == 0: ax.set_title(f"d={d}")
            if j == 0: ax.set_ylabel(f"r={r}\nLER")
            if i == nr - 1: ax.set_xlabel("p")

    axes[0, -1].legend(loc="best", fontsize=7)
    fig.suptitle("Unrotated SC — Z Injection: LER vs p", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = "summary_Z.png"
    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

CHECKPOINT_KEYS = ["inject_state", "post_select_mode", "d", "rounds", "p"]
CSV_PATH = OUT_DIR / "unrotated_sc_eval.csv"


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
    parser = argparse.ArgumentParser(description="State Injection Eval — Unrotated SC")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    sweep = QUICK_SWEEP if args.quick else FULL_SWEEP
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("State Injection Evaluation — Unrotated Surface Code")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    all_tasks = build_tasks(sweep)
    completed = load_completed_keys()
    tasks = filter_tasks(all_tasks, completed)
    print(f"Tasks: {len(tasks)} to run / {len(all_tasks)} total")

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

    # Load full dataset for plots/pivot
    df = pd.read_csv(CSV_PATH)
    sort_cols = [c for c in CHECKPOINT_KEYS if c in df.columns]
    df = df.sort_values(sort_cols, ignore_index=True)

    ler_wide = df.pivot_table(
        index=["inject_state", "rounds", "d", "p"],
        columns="post_select_mode",
        values="logical_error_rate",
    ).reset_index()
    ler_wide.to_csv(OUT_DIR / "unrotated_sc_ler_wide.csv", index=False)

    print("\nGenerating plots...")
    plot_ler_vs_p_by_mode(df, OUT_DIR)
    plot_ler_vs_p_by_state(df, OUT_DIR)
    plot_ps_rate(df, OUT_DIR)
    plot_summary_grid(df, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
