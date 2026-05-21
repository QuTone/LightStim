"""
Build a quantum memory experiment end-to-end on a rotated surface code.

A memory experiment initializes all data qubits in a logical basis (Z or X),
runs repeated syndrome extraction rounds to detect errors, then measures all
data qubits and checks if the logical observable was preserved.

This is the simplest complete workflow in LightStim and the template for
every other experiment type.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment


def main():
    # 1. Build the code patch (geometry + stabilizers + logicals)
    patch = RotatedSurfaceCode(distance=3)
    print(f"Patch: {patch.num_qubits} qubits, {len(patch.stabilizers)} stabilizers, "
          f"{patch.num_logicals} logical(s)")

    # 2. Wrap in a QECSystem (global canvas for one or more patches)
    system = QECSystem()
    system.add_patch(patch, name='main')

    # 3. Configure circuit-level depolarizing noise
    noise = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)

    # 4. Build the memory experiment (Z-basis, d rounds)
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        rounds=3,
        noise_params=noise,
        noise_model='circuit_level',
        basis='Z',
    )
    circuit = exp.build()

    # 5. Inspect the result
    print(f"Circuit: {circuit.num_qubits} qubits, "
          f"{circuit.num_detectors} detectors, "
          f"{circuit.num_observables} observable(s)")

    # Noiseless sanity check: zero detection events expected on clean circuit
    noiseless_circuit = exp.builder.circuit
    sampler = noiseless_circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(shots=10, separate_observables=True)
    assert detection_events.sum() == 0, "Unexpected detection events in noiseless circuit"
    print("Noiseless check passed: 0 detection events in 10 shots")


if __name__ == '__main__':
    main()
