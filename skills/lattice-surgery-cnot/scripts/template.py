"""
Implement a logical CNOT via lattice surgery using 3 surface code patches.

Lattice surgery merges and splits code patches by temporarily activating
joint stabilizer measurements (XX or ZZ couplers) between adjacent patches.

Layout:
        Ancilla (A)
           |  ZZ coupler
        Control (C)  —— XX coupler ——  Target (T)

Protocol (two rounds of lattice surgery):
  1. Prepare A in |+⟩; measure ZZ(C,A) and XX(T,A); measure A in Z.
  2. Prepare A in |0⟩; measure XX(T,A) and ZZ(C,A); measure A in X.

This implements CNOT: C is control, T is target.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lightstim.noise.config import NoiseConfig
from lightstim.protocols.cnot_ls import CNOTLSExperiment


def main():
    exp = CNOTLSExperiment(
        patch_configs={
            'c': {'distance': 3},   # control patch
            't': {'distance': 3},   # target patch
            'a': {'distance': 3},   # ancilla patch (consumed during surgery)
        },
        offset_ta=(8, 0),   # target placed 8 units right of ancilla
        offset_ca=(0, 8),   # control placed 8 units below ancilla
        initial_state_dict={'a': 'X', 'c': 'X', 't': 'X'},
        measure_state_dict={'a': 'Z', 'c': 'X', 't': 'X'},
        rounds=2,
        noise_params=NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3),
        noise_model='circuit_level',
    )
    circuit = exp.build()

    print(f"Lattice surgery CNOT circuit: {circuit.num_qubits} qubits, "
          f"{circuit.num_detectors} detectors, "
          f"{circuit.num_observables} observable(s)")

    # Noiseless check
    noiseless = exp.builder.circuit
    sampler = noiseless.compile_detector_sampler()
    det_events, _ = sampler.sample(shots=20, separate_observables=True)
    assert det_events.sum() == 0, "Detection events in noiseless LS-CNOT circuit"
    print("Noiseless check passed: 0 detection events over 20 shots")


if __name__ == '__main__':
    main()
