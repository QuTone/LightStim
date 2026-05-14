"""
Run a memory experiment through the simulation pipeline and read out the
logical error rate (LER).

Shows how to:
  - Build circuits at multiple (distance, noise) operating points
  - Run them through SimulationPipeline with PyMatching
  - Read SimulationStats and print LER

Swap DecoderConfig(name='pymatching') for 'bposd' or 'mwpf' to use other decoders
(requires the corresponding optional dependency to be installed).
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
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig


def build_circuit(distance: int, p: float):
    patch = RotatedSurfaceCode(distance=distance)
    system = QECSystem()
    system.add_patch(patch, name='main')
    noise = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        rounds=distance,
        noise_params=noise,
        noise_model='circuit_level',
        basis='Z',
    )
    return exp.build()


def main():
    decoder_cfg = DecoderConfig(name='pymatching')
    pipeline = SimulationPipeline(
        decoder_config=decoder_cfg,
        max_errors=100,
        print_progress=False,
    )

    operating_points = [(3, 1e-3), (3, 5e-3), (5, 1e-3), (5, 5e-3)]
    print(f"{'distance':>8}  {'p':>8}  {'LER':>10}  {'shots':>8}")
    print("-" * 42)
    for distance, p in operating_points:
        circuit = build_circuit(distance, p)
        stats = pipeline.run(circuit, json_metadata={'d': distance, 'p': p})
        print(f"{distance:>8}  {p:>8.0e}  {stats.logical_error_rate:>10.3e}  "
              f"{stats.post_selected_shots:>8}")


if __name__ == '__main__':
    main()
