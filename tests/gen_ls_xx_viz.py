"""
Generate MWPF visualization materials for d=3 two-patch XX lattice surgery.

Setup: both patches initialized in Z basis, XX measurement, measured in Z basis.

Outputs (all in tests/mwpf_viz_output/):
  ls_xx_zz_circuit.stim         — noiseless Stim circuit, rounds=2, flattened
  ls_xx_zz_circuit_level.html   — MWPF DEM visualization (circuit-level, rounds=d=3)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector

try:
    import mwpf
except ImportError:
    raise ImportError("mwpf not installed. Run: pip install mwpf frozendict frozenlist")

OUTPUT_DIR = Path(__file__).resolve().parent / "mwpf_viz_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

D       = 3
ROUNDS_VIZ = 2  # rounds for HTML visualization
P       = 1e-2
T_SCALE = 3.0

# ── Stim circuit: rounds=2, flattened ────────────────────────────────────────

print("Building Stim circuit: XX measurement, |Z>|Z> init, rounds=2, flattened...")
exp_stim = TwoPatchLSExperiment(
    patch1_config={"distance": D},
    patch2_config={"distance": D},
    offset=(6, 0),
    interaction_type="XX",
    initial_state_patch1="Z",
    initial_state_patch2="Z",
    measure_state_patch1="Z",
    measure_state_patch2="Z",
    rounds=2,
    noise_params=None,
    noise_model=None,
)
circuit_stim = exp_stim.build()
circuit_flat = circuit_stim.flattened()

print(f"  Qubits: {circuit_flat.num_qubits}, "
      f"Detectors: {circuit_flat.num_detectors}, "
      f"Observables: {circuit_flat.num_observables}, "
      f"Measurements: {circuit_flat.num_measurements}")

ds, obs = circuit_flat.compile_detector_sampler().sample(shots=10, separate_observables=True)
print(f"  Noiseless check: {'OK' if not np.any(ds) else 'FAIL'}")

stim_path = OUTPUT_DIR / "ls_xx_zz_circuit.stim"
stim_path.write_text(str(circuit_flat))
print(f"  Saved: {stim_path}")

# ── HTML visualization: rounds=d=3 ───────────────────────────────────────────

print(f"\nBuilding HTML visualization circuit: rounds={ROUNDS_VIZ}...")
exp_viz = TwoPatchLSExperiment(
    patch1_config={"distance": D},
    patch2_config={"distance": D},
    offset=(6, 0),
    interaction_type="XX",
    initial_state_patch1="Z",
    initial_state_patch2="Z",
    measure_state_patch1="Z",
    measure_state_patch2="Z",
    rounds=ROUNDS_VIZ,
    noise_params=None,
    noise_model=None,
)
circuit_viz = exp_viz.build()
print(f"  Qubits: {circuit_viz.num_qubits}, "
      f"Detectors: {circuit_viz.num_detectors}, "
      f"Observables: {circuit_viz.num_observables}")

# ── Inject circuit-level noise ────────────────────────────────────────────────

print(f"\nInjecting circuit-level noise p={P:.0e}...")
nc  = NoiseConfig(p_1q=P, p_2q=P, p_meas=P, p_reset=P, p_idle=P)
inj = NoiseInjector.from_circuit_level(nc, list(range(circuit_viz.num_qubits)))
noisy_circuit = inj.inject_noise(circuit_viz)

dem = noisy_circuit.detector_error_model(decompose_errors=True)
print(f"  DEM: {dem.num_errors} error mechanisms (after decompose_errors=True)")

# ── Sample syndrome ───────────────────────────────────────────────────────────

print("\nSampling syndrome (target=4 defects)...")
sampler = noisy_circuit.compile_detector_sampler(seed=42)
defects = None

for attempt in range(10_000):
    shot = sampler.sample(shots=1, bit_packed=False)
    d_list = list(np.where(shot[0])[0])
    if len(d_list) == 4:
        defects = d_list
        print(f"  Found 4-defect syndrome at attempt {attempt+1}: {defects}")
        break

if defects is None:
    shots = sampler.sample(shots=1000, bit_packed=False, seed=43)
    for row in shots:
        d_list = list(np.where(row)[0])
        if 2 <= len(d_list) <= 6:
            defects = d_list
            print(f"  Fallback: {len(d_list)}-defect syndrome: {defects}")
            break

if defects is None:
    defects = [0, 1]
    print(f"  WARNING: using fallback defects {defects}")

# ── MWPF solve + visualize ────────────────────────────────────────────────────

print("\nRunning MWPF solver...")
ref_dem  = mwpf.RefDetectorErrorModel.of(dem=dem)
solver   = mwpf.SolverSerialJointSingleHair(ref_dem.initializer)
syndrome = mwpf.SyndromePattern(defect_vertices=defects)

coords = circuit_viz.get_detector_coordinates()
positions = []
for det_id in range(circuit_viz.num_detectors):
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
print(f"  Snapshots: {visualizer.snapshots}")

html_path = OUTPUT_DIR / "ls_xx_zz_circuit_level.html"
html_path.write_text(visualizer.generate_html())
print(f"  Saved: {html_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n── Done ──────────────────────────────────────────────────────")
print(f"  {stim_path.name}    — noiseless circuit, rounds=2, flattened")
print(f"  {html_path.name}  — MWPF viz, rounds={ROUNDS_VIZ}, circuit-level")
print("\nBrowser tips:")
print("  T/L/F = top/left/front camera  |  ←→ step through snapshots")
print("  Config panel: set background=white, adjust node/edge sizes")
print("  Ctrl+S: export PNG")
