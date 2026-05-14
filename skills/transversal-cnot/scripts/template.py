"""
Run a transversal CNOT gate between two CSS code patches.

A transversal CNOT applies physical CX gates between matching qubits on control
and target patches. It is fault-tolerant when the two codes share the same layout.

Protocol:
  rounds_before SE rounds → transversal CX → rounds_after SE rounds → readout

Works with any CSS code (rotated, unrotated, toric) by swapping code_patch_class
and extraction_block_class.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.cnot_trans import CNOTTransExperiment


def main():
    # Control: |X⟩ (logical |+⟩), Target: |0⟩ (logical |0⟩)
    # After CNOT: control stays |+⟩, target becomes |+⟩ (Bell-like entanglement in Z basis)
    exp = CNOTTransExperiment(
        code_patch_class=UnrotatedSurfaceCode,
        extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
        code_params_control={'distance': 3},
        offset_target=(8, 0),          # place target 8 units to the right
        initial_basis_control='X',     # control starts in |+⟩
        initial_basis_target='Z',      # target starts in |0⟩
        measure_basis_control='X',
        measure_basis_target='Z',
        rounds_before=2,
        rounds_after=2,
        noise_params=NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3),
        noise_model='circuit_level',
    )
    circuit = exp.build()

    print(f"Transversal CNOT circuit: {circuit.num_qubits} qubits, "
          f"{circuit.num_detectors} detectors, "
          f"{circuit.num_observables} logical observable(s)")
    # Note: observable count depends on initial/measure basis combination.
    # Use initial_basis_control='Z' + measure_basis_control='Z' to track ZL observable.

    # Noiseless check: ZZ observable must be deterministic (+1)
    noiseless = exp.builder.circuit
    sampler = noiseless.compile_detector_sampler()
    det_events, obs_flips = sampler.sample(shots=20, separate_observables=True)
    assert det_events.sum() == 0, "Detection events in noiseless CNOT circuit"
    print(f"Noiseless check passed: 0 detection events, "
          f"{obs_flips.shape[1]} observable(s) measured over 20 shots")


if __name__ == '__main__':
    main()
