"""
Inject an arbitrary logical state (Z, X, or Y) into a surface code patch.

State injection encodes a physical qubit state into a logical qubit without a
full transversal encoding circuit. The injected state is "grown" into a full
distance-d code block through SE rounds, then read out transversally.

Post-selection modes:
  full_postselection — discard shots with any detection event (high purity, low rate)
  full_qec          — decode and correct all errors (no post-selection overhead)
  hybrid            — post-select on injection detectors, decode the rest (middle ground)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
    RotatedSurfaceCodeLogicalOpSet,
)
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.state_injection import StateInjectionExperiment


def main():
    noise = NoiseConfig(p_1q=5e-4, p_2q=5e-4, p_meas=5e-4, p_reset=5e-4, p_idle=5e-4)

    for inject_state in ['Z', 'X', 'Y']:
        exp = StateInjectionExperiment(
            code_patch_class=RotatedSurfaceCode,
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            op_set_class=RotatedSurfaceCodeLogicalOpSet,
            distance=3,
            rounds=2,
            inject_state=inject_state,
            protocol='corner',
            post_select_mode='full_postselection',
            noise_params=noise,
            noise_model='circuit_level',
        )
        circuit = exp.build()

        # Noiseless sanity check
        noiseless = exp.builder.circuit
        sampler = noiseless.compile_detector_sampler()
        det_events, obs_flips = sampler.sample(shots=50, separate_observables=True)
        n_det_errors = det_events.sum()
        n_obs_errors = obs_flips.sum()

        print(f"|{inject_state}⟩ injection: {circuit.num_detectors} detectors, "
              f"{circuit.num_observables} observable  "
              f"— noiseless: {n_det_errors} det events, {n_obs_errors} logical errors "
              f"over 50 shots")


if __name__ == '__main__':
    main()
