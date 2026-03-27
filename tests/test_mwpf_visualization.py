"""
MWPF Decoding Graph Visualization — for paper intro figure.

Generates interactive HTML visualizations of a d=3 rotated surface code
decoding graph with detection events (defects) and matched edges highlighted.

Exports two variants for comparison:
  - circuit_level noise  (realistic, ~150 edges)
  - phenomenological noise (cleaner, ~45 edges, recommended for paper)

Usage:
    python tests/test_mwpf_visualization.py

Output:
    tests/mwpf_viz_output/decoding_graph_circuit_level.html
    tests/mwpf_viz_output/decoding_graph_phenomenological.html
    → Open in browser. Use keyboard shortcuts to adjust view, then screenshot.

Key controls in browser:
    T / L / F   → Top / Left / Front camera preset
    Arrow keys  → Step through algorithm snapshots (grow → match → subgraph)
    Ctrl+S      → Export PNG directly from browser
    Config panel (top-right) → set background=white for paper, adjust node/edge size

Design notes:
    - Noise model: phenomenological recommended for intro figure
      * data qubit errors  → spatial edges  (same-round detector pairs)
      * measurement errors → temporal edges (same detector, adjacent rounds)
      * No CNOT error chains → ~3× fewer edges than circuit_level
    - Layer spacing: T_SCALE multiplies the t-coordinate so round layers are
      visually separated (default 3× → rounds at t=0,3,6 vs spatial span ~4)
    - decompose_errors=True: splits hyperedges into degree-2 pairs → cleaner graph
"""

import io
import contextlib
import sys
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
from src.ir.qec_system import QECSystem
from src.noise.config import NoiseConfig
from experiments.memory import MemoryExperiment

try:
    import mwpf
except ImportError:
    raise ImportError("mwpf not installed. Run: pip install mwpf frozendict frozenlist")

OUTPUT_DIR = Path(__file__).resolve().parent / "mwpf_viz_output"

# ── Visualization parameters ──────────────────────────────────────────────────
DISTANCE = 3
ROUNDS   = 3
P        = 1e-2   # slightly elevated → easier to sample a clean syndrome

# Layer spacing: multiply t-coordinate so rounds are visually separated.
# Spatial extent for d=3 is ~4 units; T_SCALE=3 → rounds at t=0,3,6.
T_SCALE = 3.0

TARGET_DEFECTS = 4   # number of defects to look for in the sampled syndrome
SAMPLE_SEED    = 42


# ── Circuit builders ──────────────────────────────────────────────────────────

def build_circuit(noise_model: str, p: float) -> stim.Circuit:
    """
    Build a d=DISTANCE rotated surface code memory-Z circuit.
    noise_model: "circuit_level" or "phenomenological"
    """
    code = RotatedSurfaceCode(distance=DISTANCE)
    code.rotate_coords(np.pi / 4)
    system = QECSystem()
    system.add_patch(code, name="rotated_sc")
    noise = NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p)

    with contextlib.redirect_stdout(io.StringIO()):
        exp = MemoryExperiment(
            qec_system=system,
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            rounds=ROUNDS,
            noise_params=noise,
            noise_model=noise_model,
            basis="Z",
        )
        circuit = exp.build()
    return circuit


# ── Syndrome sampling ─────────────────────────────────────────────────────────

def sample_syndrome(circuit: stim.Circuit, target: int = TARGET_DEFECTS,
                    seed: int = SAMPLE_SEED) -> list[int]:
    """
    Sample syndromes until one has exactly `target` defects.
    Falls back to any syndrome with 2-6 defects after 10k attempts.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)

    for attempt in range(10_000):
        shot = sampler.sample(shots=1, bit_packed=False)
        defects = list(np.where(shot[0])[0])
        if len(defects) == target:
            print(f"  Found {target}-defect syndrome at attempt {attempt+1}")
            return defects

    shots = sampler.sample(shots=1000, bit_packed=False, seed=seed + 1)
    for row in shots:
        defects = list(np.where(row)[0])
        if 2 <= len(defects) <= 6:
            print(f"  Fallback: {len(defects)}-defect syndrome")
            return defects

    print("  WARNING: no clean syndrome found, using [0, 1]")
    return [0, 1]


# ── Position extraction ───────────────────────────────────────────────────────

def detector_positions(circuit: stim.Circuit, t_scale: float = T_SCALE
                       ) -> list[mwpf.VisualizePosition]:
    """
    Map stim detector coordinates (x, y, round) → mwpf VisualizePosition(i, j, t).
    The t-coordinate is scaled by t_scale to spread layers apart visually.
    """
    coords = circuit.get_detector_coordinates()
    positions = []
    for det_id in range(circuit.num_detectors):
        c = coords.get(det_id, [0.0, 0.0, 0.0])
        i = float(c[0]) if len(c) > 0 else 0.0
        j = float(c[1]) if len(c) > 1 else 0.0
        t = float(c[2]) * t_scale if len(c) > 2 else 0.0
        positions.append(mwpf.VisualizePosition(i, j, t))
    return positions


# ── MWPF solve + visualize ────────────────────────────────────────────────────

def build_visualizer(circuit: stim.Circuit, defects: list[int]
                     ) -> tuple[mwpf.Visualizer, list[int]]:
    """
    Build the MWPF solver and run it with the given syndrome.
    Returns (Visualizer capturing all snapshots, matched subgraph edge indices).

    decompose_errors=True splits hyperedges into degree-2 edges — cleaner graph.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    print(f"  DEM: {dem.num_errors} error mechanisms (edges)")

    ref_dem = mwpf.RefDetectorErrorModel.of(dem=dem)
    solver  = mwpf.SolverSerialJointSingleHair(ref_dem.initializer)
    syndrome = mwpf.SyndromePattern(defect_vertices=defects)

    positions  = detector_positions(circuit)
    visualizer = mwpf.Visualizer(positions=positions)

    solver.solve(syndrome, visualizer=visualizer)
    subgraph = solver.subgraph(visualizer=visualizer)

    print(f"  Defects: {defects}")
    print(f"  Matched edges: {subgraph}")
    print(f"  Snapshots: {visualizer.snapshots}")
    return visualizer, subgraph


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    noise_models = [
        ("phenomenological", "phenomenological"),   # recommended for paper: ~76 edges
        ("circuit_level",    "circuit_level"),       # realistic but busy: ~286 edges
        ("XZ_biased",        "XZ_biased"),           # no Y errors: purely X/Z edges, no hyperedges
    ]

    # Sample syndrome from the phenomenological circuit (same defect IDs are
    # valid indices for either model since all have the same detector layout).
    print(f"Building d={DISTANCE} rotated SC, {ROUNDS} rounds for syndrome sampling...")
    ref_circuit = build_circuit("phenomenological", P)
    print(f"  {ref_circuit.num_qubits} qubits, {ref_circuit.num_detectors} detectors")

    print("Sampling syndrome...")
    defects = sample_syndrome(ref_circuit)

    for label, noise_model in noise_models:
        if noise_model == "XZ_biased" and defects == [0, 1]:
            print(f"\n── {label}: skipped (no valid syndrome sampled)")
            continue
        print(f"\n── {label} ──────────────────────────")
        circuit = build_circuit(noise_model, P)
        viz, subgraph = build_visualizer(circuit, defects)

        out_path = OUTPUT_DIR / f"decoding_graph_{label}.html"
        out_path.write_text(viz.generate_html())
        print(f"  Saved: {out_path}")

    print("\n── Done ─────────────────────────────────────────────────────")
    print("Open HTML files in browser:")
    print(f"  {OUTPUT_DIR}/decoding_graph_phenomenological.html  ← recommended for paper")
    print(f"  {OUTPUT_DIR}/decoding_graph_XZ_biased.html         ← no Y errors, diagonal edges only")
    print(f"  {OUTPUT_DIR}/decoding_graph_circuit_level.html     ← realistic but busy")
    print()
    print("Browser tips:")
    print("  Config panel (top-right): set background=white, adjust node/edge sizes")
    print("  T = top, L = left, F = front camera preset")
    print("  ← → arrow keys: step through solve snapshots")
    print("  Ctrl+S: export PNG")


if __name__ == "__main__":
    main()
