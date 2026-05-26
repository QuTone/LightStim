"""
Noise model comparison — template script.

Compares all four built-in noise strategies on the same circuit,
plus the XZ-biased model for hardware with asymmetric error rates.

Run from repo root:
    venv/bin/python skills/custom-noise/scripts/template.py
"""
import sys
sys.path.insert(0, ".")

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated import (
    RotatedSurfaceCode,
    RotatedSurfaceCodeExtractionBlock,
)
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

D = 3
ROUNDS = D
P = 1e-2

# ── Build noiseless circuit once ─────────────────────────────────────────────
system = QECSystem()
system.add_patch(RotatedSurfaceCode(distance=D), name="main")
tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
builder = CircuitBuilder(tracker, system)
se = RotatedSurfaceCodeExtractionBlock(system)

builder.write_coordinates()
builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
builder.apply_syndrome_extraction(se.circuit, rounds=ROUNDS)
builder.apply_data_readout({q: "Z" for q in system.data_indices})

# Noiseless sanity check
dets, obs = builder.circuit.compile_detector_sampler().sample(20, separate_observables=True)
assert not dets.any() and not obs.any(), "Build bug: noiseless circuit has events"

# ── Compare noise models ──────────────────────────────────────────────────────
pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_errors=100,
    print_progress=False,
)

configs = [
    # (strategy, NoiseConfig)  — only populate relevant fields per strategy
    ("circuit_level",    NoiseConfig(p_1q=P, p_2q=P, p_meas=P, p_reset=P)),
    ("phenomenological", NoiseConfig(p_idle=P, p_meas=P)),
    ("code_capacity",    NoiseConfig(p_idle=P)),
]

print(f"d={D}, p={P:.0e}")
print(f"{'strategy':>20}  {'LER':>10}  {'shots':>8}")
print("-" * 44)

for model, noise in configs:
    noisy = builder.build_noisy_circuit(noise, noise_model=model)
    stats = pipeline.run(noisy)
    print(f"{model:>20}  {stats.logical_error_rate:>10.3e}  {stats.post_selected_shots:>8}")

# ── XZ-biased noise (neutral atom / trapped ion) ──────────────────────────────
print("\nXZ-biased (eta=0.01, strongly Z-biased):")
biased_noise = NoiseInjector.compute_XZ_biased_params(
    p_1q=P, p_2q=P, p_meas=P, p_reset=P,
    eta=0.01,  # p_X / p_Z = 0.01: phase flips dominate
)
print(f"  p_1q_z = {biased_noise.custom_params.get('p_1q_z', 0):.2e}, "
      f"p_1q_x = {biased_noise.custom_params.get('p_1q_x', 0):.2e}")
noisy_biased = builder.build_noisy_circuit(biased_noise, noise_model="XZ_biased")
stats_biased = pipeline.run(noisy_biased)
print(f"  LER = {stats_biased.logical_error_rate:.3e}")

# ── Noiseless tag: suppress noise on specific phases ──────────────────────────
# Example: encoding (unitary block) is noiseless, SE rounds are noisy.
# builder.apply_unitary_block(encoding_circuit, noiseless=True)
# builder.apply_syndrome_extraction(se.circuit, rounds=ROUNDS)  # noisy
# builder.apply_syndrome_extraction(se.circuit, rounds=1, noiseless=True)  # suppress
print("\nNote: pass noiseless=True to initialize()/apply_syndrome_extraction()/")
print("apply_unitary_block() to suppress noise on specific circuit phases.")
print("The noise injector skips any instruction tagged 'noiseless'.")
