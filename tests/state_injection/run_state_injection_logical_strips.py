"""
State Injection Evaluation with logical-strip hybrid post-selection.

Compares:
- full_qec
- full_postselection
- hybrid (hybrid_post_select_scheme='logical_strips')

Sweep parameters:
- injection_protocol = ['middle', 'corner']
- inject_state       = ['Z', 'X']
- rounds             = [2, 3]
- distance           = [3, 5, 7]
- p                  = [1e-4, 5e-4, 1e-3]
"""

import contextlib
import io
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.state_injection import StateInjectionExperiment
from src.noise.config import NoiseConfig
from src.simulation.decoder_backend import DecoderConfig, ExperimentTask, SimulationPipeline

INJECTION_PROTOCOLS = ["middle", "corner"]
INJECT_STATES = ["Z", "X"]
POST_SELECT_MODES = ["full_qec", "hybrid", "full_postselection"]
DISTANCES = [3, 5, 7]
ROUNDS_LIST = [2, 3]
PER_LIST = [1e-4, 5e-4, 1e-3]

MAX_SHOTS = 1_000_000
MAX_ERRORS = 100
NUM_WORKERS = 32

PREFIX = "state_injection_logical_strips"


def build_tasks():
    tasks = []
    for protocol in INJECTION_PROTOCOLS:
        for inject_state in INJECT_STATES:
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
                                    hybrid_post_select_scheme="logical_strips",
                                    inject_state=inject_state,
                                    noise_params=noise_params,
                                    noise_model="circuit_level",
                                )
                                circuit = exp.build()
                            tasks.append(
                                ExperimentTask(
                                    circuit,
                                    json_metadata={
                                        "injection_protocol": protocol,
                                        "inject_state": inject_state,
                                        "post_select_mode": mode,
                                        "rounds": rounds,
                                        "d": d,
                                        "p": p,
                                    },
                                )
                            )
    return tasks


def run_simulation(tasks):
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching", backend="cpu"),
        max_errors=MAX_ERRORS,
        max_shots=MAX_SHOTS,
        num_workers=NUM_WORKERS,
        print_progress=True,
    )
    df = pipeline.run_batch(tasks)
    return df.sort_values(
        ["inject_state", "injection_protocol", "rounds", "d", "p", "post_select_mode"],
        ignore_index=True,
    )


def save_results(df: pd.DataFrame):
    csv_long = SCRIPT_DIR / f"{PREFIX}_results.csv"
    df.to_csv(csv_long, index=False)
    print(f"Saved: {csv_long}")

    ler_wide = (
        df.pivot_table(
            index=["inject_state", "injection_protocol", "rounds", "d", "p"],
            columns="post_select_mode",
            values="logical_error_rate",
        )
        .rename(
            columns={
                "full_qec": "ler_full_qec",
                "hybrid": "ler_hybrid",
                "full_postselection": "ler_full_postselection",
            }
        )
        .reset_index()
        .sort_values(["inject_state", "injection_protocol", "rounds", "d", "p"], ignore_index=True)
    )
    csv_ler = SCRIPT_DIR / f"{PREFIX}_ler_wide.csv"
    ler_wide.to_csv(csv_ler, index=False)
    print(f"Saved: {csv_ler}")

    ps_wide = (
        df.pivot_table(
            index=["inject_state", "injection_protocol", "rounds", "d", "p"],
            columns="post_select_mode",
            values="post_selection_rate",
        )
        .rename(
            columns={
                "full_qec": "ps_rate_full_qec",
                "hybrid": "ps_rate_hybrid",
                "full_postselection": "ps_rate_full_postselection",
            }
        )
        .reset_index()
        .sort_values(["inject_state", "injection_protocol", "rounds", "d", "p"], ignore_index=True)
    )
    csv_ps = SCRIPT_DIR / f"{PREFIX}_ps_rate_wide.csv"
    ps_wide.to_csv(csv_ps, index=False)
    print(f"Saved: {csv_ps}")


def plot_ler(df: pd.DataFrame):
    for inject_state in INJECT_STATES:
        for protocol in INJECTION_PROTOCOLS:
            for rounds in ROUNDS_LIST:
                fig, axes = plt.subplots(1, len(DISTANCES), figsize=(5 * len(DISTANCES), 4), sharey=True)
                if len(DISTANCES) == 1:
                    axes = [axes]

                for ax, d in zip(axes, DISTANCES):
                    sub = df[
                        (df["inject_state"] == inject_state)
                        & (df["injection_protocol"] == protocol)
                        & (df["rounds"] == rounds)
                        & (df["d"] == d)
                    ]
                    for mode in POST_SELECT_MODES:
                        mode_df = sub[sub["post_select_mode"] == mode].sort_values("p")
                        ax.plot(mode_df["p"], mode_df["logical_error_rate"], marker="o", label=mode)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Physical Error Rate (p)")
                    ax.set_title(f"{inject_state}, {protocol}, r={rounds}, d={d}")
                    ax.grid(True, which="both", alpha=0.3)

                axes[0].set_ylabel("Logical Error Rate (LER)")
                axes[-1].legend(loc="best")
                fig.suptitle("State Injection LER: full_qec vs hybrid(logical_strips) vs full_postselection")
                fig.tight_layout()

                fig_path = SCRIPT_DIR / f"{PREFIX}_{inject_state}_{protocol}_r{rounds}_ler.png"
                fig.savefig(fig_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved plot: {fig_path}")


def plot_post_selection_rate(df: pd.DataFrame):
    for inject_state in INJECT_STATES:
        for protocol in INJECTION_PROTOCOLS:
            for rounds in ROUNDS_LIST:
                fig, axes = plt.subplots(1, len(DISTANCES), figsize=(5 * len(DISTANCES), 4), sharey=True)
                if len(DISTANCES) == 1:
                    axes = [axes]

                for ax, d in zip(axes, DISTANCES):
                    sub = df[
                        (df["inject_state"] == inject_state)
                        & (df["injection_protocol"] == protocol)
                        & (df["rounds"] == rounds)
                        & (df["d"] == d)
                    ]
                    for mode in POST_SELECT_MODES:
                        mode_df = sub[sub["post_select_mode"] == mode].sort_values("p")
                        ax.plot(mode_df["p"], mode_df["post_selection_rate"], marker="o", label=mode)
                    ax.set_xscale("log")
                    ax.set_xlabel("Physical Error Rate (p)")
                    ax.set_ylim(0.0, 1.02)
                    ax.set_title(f"{inject_state}, {protocol}, r={rounds}, d={d}")
                    ax.grid(True, which="both", alpha=0.3)

                axes[0].set_ylabel("Post-selection Rate")
                axes[-1].legend(loc="best")
                fig.suptitle(
                    "State Injection Post-selection Rate: full_qec vs hybrid(logical_strips) vs full_postselection"
                )
                fig.tight_layout()

                fig_path = SCRIPT_DIR / f"{PREFIX}_{inject_state}_{protocol}_r{rounds}_ps_rate.png"
                fig.savefig(fig_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved plot: {fig_path}")


def main():
    print("Building tasks...")
    tasks = build_tasks()
    print(f"Built {len(tasks)} tasks")

    print("Running simulation...")
    df = run_simulation(tasks)

    print("Saving outputs...")
    save_results(df)

    print("Plotting LER...")
    plot_ler(df)

    print("Plotting post-selection rates...")
    plot_post_selection_rate(df)
    print("Done.")


if __name__ == "__main__":
    main()
