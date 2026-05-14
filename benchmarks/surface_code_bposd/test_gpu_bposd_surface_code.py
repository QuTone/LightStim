"""
Surface Code Memory Experiment with BP+OSD Decoder.

Sweeps physical error rates for multiple code distances, decodes with
the GPU BP+OSD backend (cudaq_qec / nv-qldpc-decoder), and plots
Logical Error Rate vs Physical Error Rate.

Falls back to CPU BP+OSD (stimbposd) when cudaq_qec is not available.

Usage:
    python -m tests.surface_code_bposd.test_gpu_bposd_surface_code         # auto-detect GPU/CPU
    python -m tests.surface_code_bposd.test_gpu_bposd_surface_code --gpu   # force GPU
    python -m tests.surface_code_bposd.test_gpu_bposd_surface_code --cpu   # force CPU
"""

import argparse
import io
import contextlib
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.simulation.decoder_backend import (
    SimulationPipeline,
    ExperimentTask,
    DecoderConfig,
)


def _gpu_available() -> bool:
    try:
        import cudaq_qec  # noqa: F401
        return True
    except ImportError:
        return False

# ── Parameters ────────────────────────────────────────────────────────
DISTANCES = [3, 5]
PHYSICAL_ERROR_RATES = [1e-4, 5e-4, 1e-3]
NOISE_MODEL = "circuit_level"
BASIS = "Z"
MAX_SHOTS = 1_000_000
MAX_ERRORS = 100
BATCH_SIZE = 10_000

OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_FILENAME = "ler_vs_per_gpu_bposd.png"
CSV_FILENAME = "ler_vs_per_gpu_bposd.csv"


def build_circuit(distance: int, physical_error_rate: float):
    """Build a noisy memory experiment circuit for a given (d, p)."""
    code = RotatedSurfaceCode(distance=distance)
    system = QECSystem()
    system.add_patch(code, name=f"rotated_d{distance}")

    noise_params = NoiseConfig(
        p_idle=physical_error_rate,
        p_1q=physical_error_rate,
        p_2q=physical_error_rate,
        p_meas=physical_error_rate,
        p_reset=physical_error_rate,
    )

    mem_exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        rounds=distance,  # d rounds of syndrome extraction
        noise_params=noise_params,
        noise_model=NOISE_MODEL,
        basis=BASIS,
        if_detector=True,
    )

    # Suppress verbose build output
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = mem_exp.build()
    return circuit


def _make_decoder_config(use_gpu: bool) -> DecoderConfig:
    """Create BP+OSD decoder config for GPU or CPU."""
    if use_gpu:
        return DecoderConfig(
            name="nv-qldpc-decoder",
            backend="gpu",
            params={
                "max_iterations": 1000,
                "osd_order": 10,
                "bp_method": "min_sum",
                "osd_method": "osd_cs",
                "use_osd": True,
            },
        )
    else:
        return DecoderConfig(
            name="bposd",
            backend="cpu",
            params={
                "max_iterations": 1000,
                "osd_order": 10,
                "bp_method": "min_sum",
                "osd_method": "osd_cs",
            },
        )


def run_experiment(use_gpu: bool):
    """Build all tasks, decode with BP+OSD, return results DataFrame."""
    # Build circuits
    tasks = []
    for d in DISTANCES:
        for p in PHYSICAL_ERROR_RATES:
            print(f"Building circuit: d={d}, p={p:.1e}")
            circuit = build_circuit(d, p)
            tasks.append(ExperimentTask(circuit, {"d": d, "p": p}))

    decoder_config = _make_decoder_config(use_gpu)
    backend_label = "GPU" if use_gpu else "CPU"

    pipeline = SimulationPipeline(
        decoder_config=decoder_config,
        max_shots=MAX_SHOTS,
        max_errors=MAX_ERRORS,
        batch_size=BATCH_SIZE,
        num_workers=1 if use_gpu else 48,
        print_progress=True,
    )

    print(f"\nRunning {len(tasks)} tasks with {backend_label} BP+OSD decoder...")
    df = pipeline.run_batch(tasks)
    return df, backend_label


def plot_results(df: pd.DataFrame, save_path: str, backend_label: str = "GPU"):
    """Plot LER vs PER for each code distance."""
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = ["o", "s", "^", "D", "v"]
    for i, d in enumerate(sorted(df["d"].unique())):
        subset = df[df["d"] == d].sort_values("p")
        ax.loglog(
            subset["p"],
            subset["logical_error_rate"],
            marker=markers[i % len(markers)],
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            label=f"d = {d}",
        )

    ax.set_xlabel("Physical Error Rate", fontsize=13)
    ax.set_ylabel("Logical Error Rate", fontsize=13)
    ax.set_title(f"Rotated Surface Code — {backend_label} BP+OSD Decoder", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Surface Code BP+OSD experiment")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_true", help="Force GPU backend")
    group.add_argument("--cpu", action="store_true", help="Force CPU backend")
    args = parser.parse_args()

    if args.gpu:
        use_gpu = True
    elif args.cpu:
        use_gpu = False
    else:
        use_gpu = _gpu_available()
        print(f"Auto-detected: {'GPU' if use_gpu else 'CPU'} backend")

    df, backend_label = run_experiment(use_gpu)

    # Save CSV
    csv_path = OUTPUT_DIR / CSV_FILENAME
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plot
    plot_path = OUTPUT_DIR / PLOT_FILENAME
    plot_results(df, plot_path, backend_label)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(df[["d", "p", "shots", "errors", "logical_error_rate", "seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
