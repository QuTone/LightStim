"""
Experiment 0: Compare 5-patch ZZZZZ joint measurement LER vs single-patch Z memory LER.
Both at d=3, rounds=3, p=1e-3.

Expectation: ZZZZZ LER ≈ 5× Z-memory LER (each patch contributes independently).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
    UnrotatedMultiPatchCoupler,
)
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker
from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline
from lightstim.simulation.decoder_backend.config import DecoderConfig
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector


def build_zzzzz_circuit(d, rounds):
    """Build 5-patch ZZZZZ joint measurement circuit.

    Protocol:
    1. Initialize all 5 patches in |0⟩ (Z eigenstate)
    2. Syndrome extraction (pre-coupler)
    3. Activate 5-patch coupler, init coupler data in |+⟩
    4. Syndrome extraction (with coupler)
    5. MX readout on coupler data (closes measurement chain)
    6. Deactivate coupler
    7. MZ readout on all 5 patches
    """
    center = 6.0
    patch_layout = {
        'W1': (-2, 0),
        'W3': (10, 0),
        'W2': (-2, 8),
        'W4': (10, 8),
        'W5': (-2, 16),
    }

    system = QECSystem()
    for name, offset in patch_layout.items():
        p = UnrotatedSurfaceCode(distance=d)
        p.transpose_coords()
        system.add_patch(p, name=name, offset=offset)

    num_qubits = system.num_qubits
    tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    # Init all 5 patches in |0⟩ (Z basis = default R)
    all_data = {q: 'Z' for q in system.data_indices}
    builder.initialize(init_dict=all_data, n=num_qubits)

    # Pre-coupler SE
    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    # Register 5-patch coupler (ZZZZZ) — register only when needed
    system.register_coupler(UnrotatedMultiPatchCoupler(),
        patch_names=['W1', 'W2', 'W3', 'W4', 'W5'], name='zzzzz',
        path_axis='vertical', center_axis=center)
    n = system.num_qubits
    if n > tracker.num_qubits:
        tracker.expand(n - tracker.num_qubits)
    builder.write_coordinates()

    # Activate coupler
    builder.activate_coupler('zzzzz')
    cp = system.coupler_patches['zzzzz']
    cd = sorted([system.local_to_global_map['zzzzz'][q] for q in cp.data_indices])
    builder.initialize(init_dict={q: 'X' for q in cd}, n=system.num_qubits)

    # SE with coupler active
    se2 = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se2.circuit, rounds=rounds)

    # MX readout on coupler data
    measure_coupler = {q: 'X' for q in cd}
    builder.apply_data_readout(final_measurements=measure_coupler)
    builder.deactivate_coupler('zzzzz')

    # MZ readout on all 5 patches
    measure_final = {q: 'Z' for q in system.data_indices}
    builder.apply_data_readout(final_measurements=measure_final)

    circuit = builder.circuit
    print(f"  ZZZZZ circuit: {circuit.num_qubits} qubits, {circuit.num_detectors} det, {circuit.num_observables} obs")
    dem = circuit.detector_error_model(decompose_errors=True)
    print(f"  DEM OK: {dem.num_detectors} det, {dem.num_observables} obs")
    return circuit


def build_memory_circuit(d, rounds):
    """Build single-patch Z memory experiment circuit."""
    system = QECSystem()
    p = UnrotatedSurfaceCode(distance=d)
    p.transpose_coords()
    system.add_patch(p, name='P1', offset=(0, 0))

    num_qubits = system.num_qubits
    tracker = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    data_indices = list(system.data_indices)
    init_dict = {q: 'Z' for q in data_indices}
    builder.initialize(init_dict=init_dict, n=num_qubits)

    se = UnrotatedSurfaceCodeExtractionBlock(system)
    builder.apply_syndrome_extraction(circuit_chunk=se.circuit, rounds=rounds)

    measure_final = {q: 'Z' for q in data_indices}
    builder.apply_data_readout(final_measurements=measure_final)

    circuit = builder.circuit
    print(f"  Memory circuit: {circuit.num_qubits} qubits, {circuit.num_detectors} det, {circuit.num_observables} obs")
    dem = circuit.detector_error_model(decompose_errors=True)
    print(f"  DEM OK: {dem.num_detectors} det, {dem.num_observables} obs")
    return circuit


def inject_noise(circuit, p):
    """Inject circuit-level noise."""
    noise_config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    all_qubits = set()
    for inst in circuit.flattened():
        if inst.name in ("H", "S", "S_DAG", "CX", "CZ", "R", "RX", "M", "MX"):
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    all_qubits.add(t.value)
    injector = NoiseInjector.from_circuit_level(noise_config, sorted(all_qubits))
    return injector.inject_noise(circuit)


def run_sim(noisy_circuit, max_shots=10_000_000, max_errors=500):
    """Run simulation (no post-selection)."""
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching", backend="cpu"),
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=50_000,
        num_workers=32,
        print_progress=True,
        progress_interval_sec=30.0,
    )
    return pipeline.run(noisy_circuit)


if __name__ == "__main__":
    d = 3
    rounds = 3
    p = 1e-3

    print(f"=== Building circuits: d={d}, rounds={rounds} ===\n")

    print("[1] 5-Patch ZZZZZ Joint Measurement:")
    zzzzz_circuit = build_zzzzz_circuit(d, rounds)

    print(f"\n[2] Single-Patch Z Memory:")
    memory_circuit = build_memory_circuit(d, rounds)

    print(f"\n=== Running simulations at p={p} ===\n")

    print("[1] ZZZZZ Joint Measurement:")
    noisy_zzzzz = inject_noise(zzzzz_circuit, p)
    stats_zzzzz = run_sim(noisy_zzzzz)
    ler_zzzzz = stats_zzzzz.logical_error_rate
    print(f"  ZZZZZ LER = {ler_zzzzz:.4e} ({stats_zzzzz.errors} errors / {stats_zzzzz.shots} shots)\n")

    print("[2] Z Memory:")
    noisy_memory = inject_noise(memory_circuit, p)
    stats_memory = run_sim(noisy_memory)
    ler_memory = stats_memory.logical_error_rate
    print(f"  Memory LER = {ler_memory:.4e} ({stats_memory.errors} errors / {stats_memory.shots} shots)\n")

    ratio = ler_zzzzz / ler_memory if ler_memory > 0 else float('inf')
    print(f"=== Results Summary ===")
    print(f"  ZZZZZ LER:  {ler_zzzzz:.4e}")
    print(f"  Memory LER: {ler_memory:.4e}")
    print(f"  Ratio:      {ratio:.2f}x (expected ~5x)")
