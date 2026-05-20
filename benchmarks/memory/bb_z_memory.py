"""BB code [[144,12,12]] Z-basis memory with Z-only detectors, p=1e-3."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig


def main():
    p          = 1e-4
    num_rounds = 6

    system = QECSystem()
    system.add_patch(
        BBCode(l=6, m=6, A=[[3, 0], [0, 1], [0, 2]],
                           B=[[0, 3], [1, 0], [2, 0]], d=num_rounds),
        name='main',
    )

    noise = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=BBCodeExtractionBlock,
        rounds=num_rounds,
        noise_params=noise,
        noise_model='circuit_level',
        basis='Z',
        z_only=True, # circuit will only contains Z detectors
    )
    circuit = exp.build()
    flat = circuit.flattened()
    print(f"qubits={circuit.num_qubits}  detectors={circuit.num_detectors}  "
          f"obs={circuit.num_observables}  mechanisms={len(flat.detector_error_model())}")

    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(name='bposd', backend='cpu'),
        max_errors=100,
        print_progress=True,
        num_workers=10,
        max_shots=1e8,
    )
    stats = pipeline.run(circuit, json_metadata={'p': p, 'rounds': num_rounds})
    eb = stats.ler_error_bar()
    print(f"\nLER (bposd, p={p:.0e}): {stats.logical_error_rate:.3e} ± {eb:.3e}  "
          f"({stats.errors} errors / {stats.post_selected_shots} shots)")
    print(f"LER per round:          {stats.logical_error_rate / num_rounds:.3e} ± {eb / num_rounds:.3e}")


if __name__ == '__main__':
    main()
