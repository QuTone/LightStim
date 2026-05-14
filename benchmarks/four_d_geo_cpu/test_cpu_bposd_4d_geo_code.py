"""
4D Geometric Code [[18,6,3]] Memory Experiment with CPU BP+OSD Decoder.

Sweeps physical error rates with circuit-level noise model,
decodes with CPU BP+OSD (stimbposd), and plots
Logical Error Rate vs Physical Error Rate.

Usage:
    python -m tests.four_d_geo_cpu.test_cpu_bposd_4d_geo_code
"""

import io
import contextlib
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lightstim.qec_code.four_d_geo_code import FourDGeoCode, FourDGeoCodeExtractionBlock
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.simulation.decoder_backend import (
    SimulationPipeline,
    ExperimentTask,
    DecoderConfig,
)

# ── Parameters ────────────────────────────────────────────────────────
L_DET3 = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 3]]
PHYSICAL_ERROR_RATES = [1e-4, 5e-4, 1e-3]
NOISE_MODEL = "circuit_level"
BASIS = "Z"
ROUNDS = 3  # d=3 for Det3
MAX_SHOTS = 1_000_000
MAX_ERRORS = 100
BATCH_SIZE = 10_000

OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_FILENAME = "ler_vs_per_cpu_bposd_4d_geo.png"
CSV_FILENAME = "ler_vs_per_cpu_bposd_4d_geo.csv"


def build_circuit(physical_error_rate: float):
    """Build a noisy memory experiment circuit for Det3 [[18,6,3]]."""
    code = FourDGeoCode(L=L_DET3)
    system = QECSystem()
    system.add_patch(code, name="det3")

    noise_params = NoiseConfig(
        p_idle=physical_error_rate,
        p_1q=physical_error_rate,
        p_2q=physical_error_rate,
        p_meas=physical_error_rate,
        p_reset=physical_error_rate,
    )

    mem_exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=FourDGeoCodeExtractionBlock,
        rounds=ROUNDS,
        noise_params=noise_params,
        noise_model=NOISE_MODEL,
        basis=BASIS,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        circuit = mem_exp.build()
    return circuit


def run_experiment():
    """Build all tasks, decode with CPU BP+OSD, return results DataFrame."""
    tasks = []
    for p in PHYSICAL_ERROR_RATES:
        print(f"Building circuit: Det3 [[18,6,3]], p={p:.1e}")
        circuit = build_circuit(p)
        tasks.append(ExperimentTask(circuit, {"p": p}))

    decoder_config = DecoderConfig(
        name="bposd",
        backend="cpu",
        params={
            "max_iterations": 1000,
            "osd_order": 10,
            "bp_method": "min_sum",
            "osd_method": "osd_cs",
        },
    )

    pipeline = SimulationPipeline(
        decoder_config=decoder_config,
        max_shots=MAX_SHOTS,
        max_errors=MAX_ERRORS,
        batch_size=BATCH_SIZE,
        num_workers=4,
        print_progress=True,
    )

    print(f"\nRunning {len(tasks)} tasks with CPU BP+OSD decoder...")
    df = pipeline.run_batch(tasks)
    return df


def plot_results(df: pd.DataFrame, save_path: str):
    """Plot LER vs PER for the Det3 code."""
    fig, ax = plt.subplots(figsize=(8, 6))

    df_sorted = df.sort_values("p")
    ax.loglog(
        df_sorted["p"],
        df_sorted["logical_error_rate"],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=8,
        color="#1f77b4",
        label="Det3 [[18,6,3]]",
    )

    ax.set_xlabel("Physical Error Rate", fontsize=13)
    ax.set_ylabel("Logical Error Rate", fontsize=13)
    ax.set_title("4D Geometric Code [[18,6,3]] — CPU BP+OSD Decoder", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close(fig)


def main():
    df = run_experiment()

    # Save CSV
    csv_path = OUTPUT_DIR / CSV_FILENAME
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plot
    plot_path = OUTPUT_DIR / PLOT_FILENAME
    plot_results(df, plot_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary: 4D Geometric Code [[18,6,3]] Memory Experiment")
    print("=" * 60)
    print(df[["p", "shots", "errors", "logical_error_rate", "seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
