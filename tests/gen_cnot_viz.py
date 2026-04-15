"""
Generate MWPF visualization for d=3 CNOT-LS circuit.

Setup (notebook case 3):
  Control init=X, Target init=Z, Ancilla init=X
  Measure: Control=X, Target=X, Ancilla=Z
  rounds=2, circuit-level noise p=1e-2

Outputs in tests/mwpf_viz_output/:
  cnot_ls_circuit.stim       — noiseless Stim circuit, rounds=2, flattened
  cnot_ls_circuit_level.html — MWPF DEM visualization
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.CNOT_LS import CNOTLSExperiment
from src.qec_code.surface_code.unrotated import UnrotatedSurfaceCodeExtractionBlock
from src.noise.config import NoiseConfig
from src.noise.injector import NoiseInjector

try:
    import mwpf
except ImportError:
    raise ImportError("mwpf not installed.")

OUTPUT_DIR = Path(__file__).resolve().parent / "mwpf_viz_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

P       = 1e-2
T_SCALE = 3.0
D       = 3
ROUNDS  = 1

patch_configs = {"c": {"distance": D}, "t": {"distance": D}, "a": {"distance": D}}

# ── Noiseless circuit (rounds=2, flattened) ───────────────────────────────────

print(f"Building CNOT-LS circuit: d={D}, rounds={ROUNDS}, ctrl=X, tgt=Z, meas=X/X...")
exp = CNOTLSExperiment(
    patch_configs=patch_configs,
    offset_ta=(6, 0),
    offset_ca=(0, 6),
    initial_state_dict={"a": "X", "c": "X", "t": "Z"},
    measure_state_dict={"a": "Z", "c": "X", "t": "X"},
    extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
    rounds=ROUNDS,
    noise_params=None,
    noise_model=None,
)
clean = exp.build()
flat  = clean.flattened()
print(f"  Qubits: {flat.num_qubits}, Detectors: {flat.num_detectors}, "
      f"Observables: {flat.num_observables}, Measurements: {flat.num_measurements}")

ds, obs = flat.compile_detector_sampler().sample(shots=10, separate_observables=True)
print(f"  Noiseless check: {'OK' if not np.any(ds) else 'FAIL'}")

stim_path = OUTPUT_DIR / "cnot_ls_circuit.stim"
stim_path.write_text(str(flat))
print(f"  Saved: {stim_path}")

# ── Inject circuit-level noise ────────────────────────────────────────────────

print(f"\nInjecting circuit-level noise p={P:.0e}...")
nc  = NoiseConfig(p_1q=P, p_2q=P, p_meas=P, p_reset=P, p_idle=P)
inj = NoiseInjector.from_circuit_level(nc, list(range(clean.num_qubits)))
noisy = inj.inject_noise(clean)

dem = noisy.detector_error_model(decompose_errors=True)
print(f"  DEM: {dem.num_errors} error mechanisms")

# ── Sample syndrome ───────────────────────────────────────────────────────────

print("\nSampling syndrome (target 4–6 defects)...")
defects = None
for seed in range(20):
    sampler = noisy.compile_detector_sampler(seed=seed)
    shots = sampler.sample(shots=500, bit_packed=False)
    for row in shots:
        d_list = list(np.where(row)[0])
        if 4 <= len(d_list) <= 6:
            defects = d_list
            print(f"  Found {len(d_list)}-defect syndrome (seed={seed}): {defects}")
            break
    if defects is not None:
        break

if defects is None:
    defects = [0, 1]
    print(f"  WARNING: using fallback defects {defects}")

# ── MWPF solve + visualize ────────────────────────────────────────────────────

print("\nRunning MWPF solver...")
ref_dem  = mwpf.RefDetectorErrorModel.of(dem=dem)
solver   = mwpf.SolverSerialJointSingleHair(ref_dem.initializer)
syndrome = mwpf.SyndromePattern(defect_vertices=defects)

coords = clean.get_detector_coordinates()
positions = []
for det_id in range(clean.num_detectors):
    c = coords.get(det_id, [0.0, 0.0, 0.0])
    i = float(c[0]) if len(c) > 0 else 0.0
    j = float(c[1]) if len(c) > 1 else 0.0
    t = float(c[2]) * T_SCALE if len(c) > 2 else 0.0
    positions.append(mwpf.VisualizePosition(i, j, t))

visualizer = mwpf.Visualizer(positions=positions)
solver.solve(syndrome, visualizer=visualizer)
subgraph   = solver.subgraph(visualizer=visualizer)
print(f"  Defects: {defects}")
print(f"  Matched subgraph edges: {subgraph}")
print(f"  Snapshots: {len(visualizer.snapshots)}")

html_path = OUTPUT_DIR / "cnot_ls_circuit_level.html"
html_path.write_text(visualizer.generate_html())
print(f"  Saved: {html_path}")

print("\n── Done ──────────────────────────────────────────────────────")
print(f"  {stim_path.name}      — noiseless circuit, rounds=2, flattened")
print(f"  {html_path.name}  — MWPF viz, circuit-level")
