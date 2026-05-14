"""
Configure and compare different noise models on a memory experiment.

LightStim supports three built-in noise model strategies:
  circuit_level    — depolarizing noise after every gate + measurement flip
  phenomenological — only measurement errors + data errors between rounds
  code_capacity    — data errors only (no gate/measurement errors), ideal model

NoiseConfig is a dataclass. Standard fields (p_1q, p_2q, p_meas, p_reset, p_idle)
cover most use cases. Unusual rates (e.g. biased noise) go in custom_params.
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


def build(noise_model: str, noise: NoiseConfig):
    patch = RotatedSurfaceCode(distance=3)
    system = QECSystem()
    system.add_patch(patch, name='main')
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=RotatedSurfaceCodeExtractionBlock,
        rounds=3,
        noise_params=noise,
        noise_model=noise_model,
        basis='Z',
    )
    return exp.build()


def main():
    p = 1e-2
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig(name='pymatching'),
        max_errors=100,
        print_progress=False,
    )

    configs = [
        ('circuit_level',    NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)),
        ('phenomenological', NoiseConfig(p_meas=p, p_idle=p)),
        ('code_capacity',    NoiseConfig(p_idle=p)),
    ]

    print(f"{'model':>20}  {'LER':>10}  {'shots':>8}")
    print("-" * 44)
    for model_name, noise_cfg in configs:
        circuit = build(model_name, noise_cfg)
        stats = pipeline.run(circuit, json_metadata={'model': model_name})
        print(f"{model_name:>20}  {stats.logical_error_rate:>10.3e}  "
              f"{stats.post_selected_shots:>8}")

    # Custom params example: asymmetric T1/T2 noise
    custom_noise = NoiseConfig(
        p_2q=p,
        p_meas=p,
        custom_params={'p_z': p * 10, 'p_x': p * 0.1},  # Z-biased idle noise
    )
    print(f"\nCustom params: p_z={custom_noise.custom_params['p_z']:.0e}, "
          f"p_x={custom_noise.custom_params['p_x']:.0e}")
    print("  (use noise.get('p_z') in a custom NoiseInjector rule to apply these)")


if __name__ == '__main__':
    main()
