"""
4D Hadamard [[96,6,8]] Memory Experiment with GPU BP+OSD Decoder.

Circuit-level noise, p=1e-3, 2 GPUs, max_shots=1.25e7, max_errors=25.
Paper reference LER ~2e-6.

Usage:
    python -m tests.four_d_hadamard_gpu.test_4d_hadamard_gpu_bposd
"""

import io
import contextlib
import time
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lightstim.qec_code.four_d_geo_code import FourDGeoCode, FourDGeoCodeExtractionBlock
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

# ── Parameters ────────────────────────────────────────────────────────
L_MATRIX = [[1, 1, 1, 1], [0, 2, 0, 2], [0, 0, 2, 2], [0, 0, 0, 4]]  # [[96,6,8]]
CODE_DISTANCE = 8
PHYSICAL_ERROR_RATE = 1e-3
NOISE_MODEL = "circuit_level"
BASIS = "Z"
ROUNDS = CODE_DISTANCE

MAX_SHOTS = 12_500_000
MAX_ERRORS = 25
BATCH_SIZE = 10_000
NUM_WORKERS = 2  # 2 GPUs

OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_FILENAME = "4d_hadamard_gpu_bposd.png"
CSV_FILENAME = "4d_hadamard_gpu_bposd.csv"
# ──────────────────────────────────────────────────────────────────────


def build_circuit():
    """Build the noisy memory experiment circuit for [[96,6,8]]."""
    print("Building 4D Hadamard [[96,6,8]] code...")
    t0 = time.time()
    code = FourDGeoCode(L=L_MATRIX, d=CODE_DISTANCE)
    info = code.get_info()
    print(f"  Code: [[{info['n_data']},{info['k']},{info['code_distance']}]]")
    print(f"  X-syndromes: {info['num_x_syndromes']}, Z-syndromes: {info['num_z_syndromes']}")
    print(f"  Total qubits: {len(code.qubit_coords)}")
    print(f"  Code built in {time.time() - t0:.1f}s")

    system = QECSystem()
    system.add_patch(code, name="4d_hadamard")

    noise_params = NoiseConfig(
        p_idle=PHYSICAL_ERROR_RATE,
        p_1q=PHYSICAL_ERROR_RATE,
        p_2q=PHYSICAL_ERROR_RATE,
        p_meas=PHYSICAL_ERROR_RATE,
        p_reset=PHYSICAL_ERROR_RATE,
    )

    print(f"Building memory experiment (rounds={ROUNDS}, basis={BASIS})...")
    t1 = time.time()
    mem_exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=FourDGeoCodeExtractionBlock,
        rounds=ROUNDS,
        noise_params=noise_params,
        noise_model=NOISE_MODEL,
        basis=BASIS,
        if_detector=True,
    )
    circuit = mem_exp.build()
    print(f"  Circuit: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors, "
          f"{circuit.num_observables} observables")
    print(f"  Circuit built in {time.time() - t1:.1f}s")
    return circuit


def run_experiment(circuit):
    """Decode with GPU BP+OSD using 2 GPUs."""
    decoder_config = DecoderConfig(
        name="nv-qldpc-decoder",
        backend="gpu",
        params={
            "max_iterations": 100,
            "osd_order": 10,
            "bp_method": "min_sum",
            "osd_method": "osd_cs",
            "use_osd": True,
        },
    )

    pipeline = SimulationPipeline(
        decoder_config=decoder_config,
        max_shots=MAX_SHOTS,
        max_errors=MAX_ERRORS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        print_progress=True,
    )

    print(f"\nDecoding with GPU BP+OSD ({NUM_WORKERS} GPUs)...")
    print(f"  max_shots={MAX_SHOTS:,}, max_errors={MAX_ERRORS}, batch_size={BATCH_SIZE:,}")
    stats = pipeline.run(circuit, {
        "code": "4D_Hadamard_96_6_8",
        "p": PHYSICAL_ERROR_RATE,
        "rounds": ROUNDS,
        "noise_model": NOISE_MODEL,
    })
    return stats


def plot_results(stats, save_path):
    """Simple bar/text summary plot."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    text = (
        f"4D Hadamard [[96,6,8]] — GPU BP+OSD\n"
        f"Circuit-level noise, p = {PHYSICAL_ERROR_RATE}\n"
        f"Rounds = {ROUNDS}, Basis = {BASIS}\n\n"
        f"Shots:           {stats.shots:>12,}\n"
        f"Logical errors:  {stats.errors:>12}\n"
        f"LER:             {stats.logical_error_rate:>12.2e}\n"
        f"Wall time:       {stats.seconds:>12.1f}s ({stats.seconds/3600:.2f}h)\n"
        f"Throughput:      {stats.shots/stats.seconds:>12.0f} shots/s\n"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=13,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close(fig)


def main():
    circuit = build_circuit()
    stats = run_experiment(circuit)

    # Save CSV
    csv_path = OUTPUT_DIR / CSV_FILENAME
    row = {
        "code": "4D_Hadamard_96_6_8",
        "n": 96, "k": 6, "d": CODE_DISTANCE,
        "p": PHYSICAL_ERROR_RATE,
        "rounds": ROUNDS,
        "noise_model": NOISE_MODEL,
        "decoder": "nv-qldpc-decoder (GPU BP+OSD)",
        "shots": stats.shots,
        "errors": stats.errors,
        "logical_error_rate": stats.logical_error_rate,
        "seconds": stats.seconds,
    }
    df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plot
    plot_path = OUTPUT_DIR / PLOT_FILENAME
    plot_results(stats, plot_path)

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Shots:    {stats.shots:,}")
    print(f"  Errors:   {stats.errors}")
    print(f"  LER:      {stats.logical_error_rate:.2e}")
    print(f"  Time:     {stats.seconds:.1f}s ({stats.seconds/3600:.2f}h)")
    print(f"  Rate:     {stats.shots/stats.seconds:.0f} shots/s")


if __name__ == "__main__":
    main()
