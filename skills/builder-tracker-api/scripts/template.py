"""
Builder-Tracker Direct API — template script.

Demonstrates: custom two-patch ZZ joint measurement (one round of lattice surgery)
built from scratch using CircuitBuilder + SyndromeTracker directly,
without any pre-built experiment class.

Run from repo root:
    venv/bin/python skills/builder-tracker-api/scripts/template.py
"""
import sys
sys.path.insert(0, ".")

import stim
import numpy as np

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.qec_code.surface_code.unrotated.two_patch_coupler import UnrotatedTwoPatchCoupler
from lightstim.noise.config import NoiseConfig

# ── Parameters ────────────────────────────────────────────────────────────────
D = 3
ROUNDS = D
P = 1e-3
NOISE_MODEL = "circuit_level"

# ── 1. Build system ───────────────────────────────────────────────────────────
system = QECSystem()

# ZZ interaction → patches stacked vertically (offset along y, same x-range).
# XX interaction → patches side-by-side horizontally (offset along x, same y-range).
# Unrotated SC d=3 spans y ∈ [0, 6], so offset second patch at y=8 for ZZ.
p_ctrl = system.add_patch(UnrotatedSurfaceCode(distance=D), name="ctrl", offset=(0, 0))
p_tgt  = system.add_patch(UnrotatedSurfaceCode(distance=D), name="tgt",  offset=(0, 8))

# Register a ZZ coupler between them (inactive at construction time)
coupler_protocol = UnrotatedTwoPatchCoupler()
system.register_coupler(coupler_protocol, ["ctrl", "tgt"], name="zz_coupler",
                        interaction_type="ZZ")

print(f"System: {system.num_qubits} qubits, {system.num_logicals} logicals")

# ── 2. Tracker + Builder ──────────────────────────────────────────────────────
tracker = SyndromeTracker(
    num_qubits=system.num_qubits,
    expected_num_logicals=system.num_logicals,
)
builder = CircuitBuilder(tracker, system)

# ── 3. Build SE block (reads from system.active_stabilizers each time) ────────
def make_se(system):
    return UnrotatedSurfaceCodeExtractionBlock(system)

# ── 4. Circuit: init → SE (no coupler) → activate → SE (with coupler) → readout
builder.write_coordinates()

# Phase A: initialize both patches in |0⟩ (Z basis).
# system.data_indices includes coupler data qubits even before activation,
# so filter them out — they are initialized separately in Phase C.
patch_data = {q: "Z" for q in system.data_indices
              if system.index_to_owner_map.get(q) != "zz_coupler"}
builder.initialize(patch_data, n=system.num_qubits)

# Phase B: SE rounds with patches only
se = make_se(system)
builder.apply_syndrome_extraction(se.circuit, rounds=ROUNDS)

# Phase C: Activate ZZ coupler.
# Pauses boundary stabilizers of both patches; activates joint ZZ stabilizers
# spanning the corridor between them.
builder.activate_coupler("zz_coupler")
cp = system.coupler_patches["zz_coupler"]
cp_data = {
    system.local_to_global_map["zz_coupler"][q]: "X"
    for q in cp.data_indices
}
builder.initialize(cp_data, n=system.num_qubits)

# Phase D: SE rounds with coupler active
se_coupled = make_se(system)
builder.apply_syndrome_extraction(se_coupled.circuit, rounds=ROUNDS)

# Phase E: Final readout — measure all data qubits (patches in Z, coupler in X)
builder.apply_data_readout({**patch_data, **cp_data})

# ── 5. Noiseless sanity check ─────────────────────────────────────────────────
print("\nNoiseless check...")
sampler = builder.circuit.compile_detector_sampler()
dets, obs = sampler.sample(shots=100, separate_observables=True)
assert not dets.any(), f"Detection events in noiseless circuit! Bug in build."
assert not obs.any(),  f"Observable flips in noiseless circuit! Bug in build."
print(f"  OK — 0 detection events, 0 observable flips")
print(f"  Circuit: {builder.circuit.num_qubits} qubits, "
      f"{builder.circuit.num_detectors} detectors, "
      f"{builder.circuit.num_observables} observables")

# ── 6. Add noise and simulate ─────────────────────────────────────────────────
noisy = builder.build_noisy_circuit(
    NoiseConfig(p_2q=P, p_meas=P),
    noise_model=NOISE_MODEL,
)

from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

stats = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_errors=200,
    max_shots=500_000,
    print_progress=False,
).run(noisy)

print(f"\nSimulation (p={P}, noise_model={NOISE_MODEL}):")
print(f"  LER  = {stats.logical_error_rate:.4e} ± {stats.ler_error_bar():.1e}")
print(f"  Shots = {stats.shots}, Errors = {stats.errors}")
