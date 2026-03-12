"""
State Injection Evaluation: Full QEC vs Full Postselection vs Hybrid

Sweep parameters:
- injection_protocol = ['middle', 'corner']
- post_select_mode   = ['full_qec', 'hybrid', 'full_postselection']
- distance           = [3, 5, 7]
- rounds             = [2, 3]
- p                  = [1e-4, 5e-4, 1e-3]
- inject_state       = 'Z'
- noise_model        = 'circuit_level'

Stopping: MAX_SHOTS = 1_000_000, MAX_ERRORS = 100
"""

import sys
import os
import io
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── project path ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.state_injection import StateInjectionExperiment
from src.noise.config import NoiseConfig
from src.simulation.decoder_backend import (
    SimulationPipeline,
    ExperimentTask,
    DecoderConfig,
)

# ── sweep grid ──
INJECTION_PROTOCOLS = ["middle", "corner"]
POST_SELECT_MODES = ["full_qec", "hybrid", "full_postselection"]
DISTANCES = [3, 5, 7]
ROUNDS_LIST = [2, 3]
PER_LIST = [1e-4, 5e-4, 1e-3]

MAX_SHOTS = 1_000_000
MAX_ERRORS = 100
NUM_WORKERS = 32

OUT_DIR = SCRIPT_DIR  # save results next to this script


def build_tasks():
    tasks = []
    for protocol in INJECTION_PROTOCOLS:
        for rounds in ROUNDS_LIST:
            for d in DISTANCES:
                for p in PER_LIST:
                    for mode in POST_SELECT_MODES:
                        noise_params = NoiseConfig(
                            p_idle=p, p_meas=p, p_reset=p, p_1q=p, p_2q=p
                        )
                        with contextlib.redirect_stdout(io.StringIO()):
                            exp = StateInjectionExperiment(
                                distance=d,
                                rounds=rounds,
                                injection_protocol=protocol,
                                post_select_mode=mode,
                                inject_state="Z",
                                noise_params=noise_params,
                                noise_model="circuit_level",
                            )
                            circuit = exp.build()
                        tasks.append(
                            ExperimentTask(
                                circuit,
                                json_metadata={
                                    "injection_protocol": protocol,
                                    "post_select_mode": mode,
                                    "rounds": rounds,
                                    "d": d,
                                    "p": p,
                                },
                            )
                        )
    return tasks


def run_simulation(tasks):
    decoder_config = DecoderConfig("pymatching", backend="cpu")
    pipeline = SimulationPipeline(
        decoder_config=decoder_config,
        max_errors=MAX_ERRORS,
        max_shots=MAX_SHOTS,
        num_workers=NUM_WORKERS,
        print_progress=True,
    )
    df = pipeline.run_batch(tasks)
    df = df.sort_values(
        ["injection_protocol", "rounds", "d", "p", "post_select_mode"],
        ignore_index=True,
    )
    return df


def save_results(df):
    # Long CSV
    csv_long = OUT_DIR / "state_injection_results.csv"
    df.to_csv(csv_long, index=False)
    print(f"Saved: {csv_long}")

    # LER wide
    ler_wide = (
        df.pivot_table(
            index=["injection_protocol", "rounds", "d", "p"],
            columns="post_select_mode",
            values="logical_error_rate",
        )
        .rename(columns={
            "full_qec": "ler_full_qec",
            "hybrid": "ler_hybrid",
            "full_postselection": "ler_full_postselection",
        })
        .reset_index()
        .sort_values(["injection_protocol", "rounds", "d", "p"], ignore_index=True)
    )
    csv_ler = OUT_DIR / "state_injection_ler_wide.csv"
    ler_wide.to_csv(csv_ler, index=False)
    print(f"Saved: {csv_ler}")

    # Post-selection rate wide
    ps_wide = (
        df.pivot_table(
            index=["injection_protocol", "rounds", "d", "p"],
            columns="post_select_mode",
            values="post_selection_rate",
        )
        .rename(columns={
            "full_qec": "ps_rate_full_qec",
            "hybrid": "ps_rate_hybrid",
            "full_postselection": "ps_rate_full_postselection",
        })
        .reset_index()
        .sort_values(["injection_protocol", "rounds", "d", "p"], ignore_index=True)
    )
    csv_ps = OUT_DIR / "state_injection_ps_rate_wide.csv"
    ps_wide.to_csv(csv_ps, index=False)
    print(f"Saved: {csv_ps}")


def plot_results(df):
    for protocol in INJECTION_PROTOCOLS:
        for rounds in ROUNDS_LIST:
            fig, axes = plt.subplots(
                1, len(DISTANCES), figsize=(5 * len(DISTANCES), 4), sharey=True
            )
            if len(DISTANCES) == 1:
                axes = [axes]

            for ax, d in zip(axes, DISTANCES):
                sub = df[
                    (df["injection_protocol"] == protocol)
                    & (df["rounds"] == rounds)
                    & (df["d"] == d)
                ]
                for mode in POST_SELECT_MODES:
                    mode_df = sub[sub["post_select_mode"] == mode].sort_values("p")
                    ax.plot(
                        mode_df["p"],
                        mode_df["logical_error_rate"],
                        marker="o",
                        label=mode,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{protocol}, rounds={rounds}, d={d}")
                ax.set_xlabel("Physical Error Rate (p)")
                ax.grid(True, which="both", alpha=0.3)

            axes[0].set_ylabel("Logical Error Rate (LER)")
            axes[-1].legend(loc="best")
            fig.suptitle(
                "State Injection (inject_state=Z): full_qec vs hybrid vs full_postselection"
            )
            fig.tight_layout()

            fig_path = OUT_DIR / f"state_injection_{protocol}_r{rounds}_ler.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {fig_path}")


def main():
    print("Building tasks...")
    tasks = build_tasks()
    print(f"Built {len(tasks)} tasks")

    print("Running simulation...")
    df = run_simulation(tasks)

    print("Saving results...")
    save_results(df)

    print("Plotting...")
    plot_results(df)

    print("Done!")


if __name__ == "__main__":
    main()
