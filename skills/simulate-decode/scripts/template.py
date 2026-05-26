"""
Simulate and decode — template script.

Runs a threshold sweep (LER vs p, multiple distances) on a rotated surface code.
Demonstrates: SimulationPipeline, DecoderConfig, SimulationStats, threshold crossings.

Run from repo root:
    venv/bin/python skills/simulate-decode/scripts/template.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig


def build_circuit(distance: int, p: float, basis: str = "Z") -> "stim.Circuit":
    """Build a noisy Z-memory circuit using the IR layer directly."""
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=distance), name="main")

    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    se = RotatedSurfaceCodeExtractionBlock(system)

    builder.write_coordinates()
    builder.initialize({q: basis for q in system.data_indices}, n=system.num_qubits)
    builder.apply_syndrome_extraction(se.circuit, rounds=distance)
    builder.apply_data_readout({q: basis for q in system.data_indices})

    return builder.build_noisy_circuit(
        NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p),
        noise_model="circuit_level",
    )


def main():
    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_errors=200,
        max_shots=500_000,
        print_progress=False,
    )

    # ── Threshold sweep ───────────────────────────────────────────────────────
    distances = [3, 5, 7]
    p_values  = np.logspace(-3, -1.3, 8)

    print(f"{'d':>4}  {'p':>8}  {'LER':>10}  {'±':>10}  {'shots':>8}")
    print("-" * 50)

    for d in distances:
        for p in p_values:
            circuit = build_circuit(distance=d, p=p)
            stats = pipeline.run(circuit)
            print(f"{d:>4}  {p:>8.2e}  {stats.logical_error_rate:>10.3e}  "
                  f"{stats.ler_error_bar():>10.1e}  {stats.post_selected_shots:>8}")
        print()

    # ── Reading stats ─────────────────────────────────────────────────────────
    circuit = build_circuit(distance=5, p=1e-3)
    stats = pipeline.run(circuit)

    print("stats fields:")
    print(f"  shots                = {stats.shots}")
    print(f"  post_selected_shots  = {stats.post_selected_shots}")
    print(f"  errors               = {stats.errors}")
    print(f"  logical_error_rate   = {stats.logical_error_rate:.4e}")
    print(f"  ler_error_bar(z=1.96)= {stats.ler_error_bar():.2e}  (95% Wilson CI)")
    print(f"  ler_error_bar(z=1.0) = {stats.ler_error_bar(z=1.0):.2e}  (1-sigma)")
    print(f"  post_selection_rate  = {stats.post_selection_rate:.3f}")


if __name__ == "__main__":
    main()
