"""
Extend with a new QEC code — template script.

Implements a minimal end-to-end example: a single-row repetition code
called "BitFlipStrip" that corrects bit-flip errors.

Demonstrates:
  - QECPatch subclass (geometry + stabilizers + logicals)
  - Extraction block (one SE round as stim.Circuit)
  - Verification with CircuitBuilder directly (no MemoryExperiment)
  - Noiseless sanity check

Run from repo root:
    venv/bin/python skills/extend-new-code/scripts/template.py
"""
import sys
sys.path.insert(0, ".")

import stim
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig


# ── 1. Code Patch ─────────────────────────────────────────────────────────────

class BitFlipStrip(QECPatch):
    """
    Distance-d repetition code (Z-stabilizers only, corrects bit-flip errors).

    Layout along x-axis:
      D  S  D  S  D  ...  D
      0  1  2  3  4       2d-2
    D = data (even x), S = syndrome_z (odd x)
    """

    def _process_params(self):
        self.distance = self.params.get('distance', 3)
        if self.distance < 2:
            raise ValueError("distance must be >= 2")

    def build(self):
        d = self.distance

        # Phase 1: geometry — register every qubit
        for x in range(2 * d - 1):
            role = 'data' if x % 2 == 0 else 'syndrome_z'
            self.add_qubit(x, 0, role=role)

        # Phase 2: physics — ZZ stabilizer per syndrome qubit
        # create_stim_stabilizer looks up syn_coord via self.index_map
        # → must call add_qubit first for the ancilla
        for syn_coord in self.syndrome_coords:
            left  = (syn_coord[0] - 1, 0.0)
            right = (syn_coord[0] + 1, 0.0)
            self.create_stim_stabilizer(
                {left: 'Z', right: 'Z'},
                syn_coord=syn_coord,
                type='Z',
            )

        # Phase 3: logical operators
        # Z logical: one data qubit at one end (minimum weight)
        self.create_stim_logical({(0.0, 0.0): 'Z'}, op_type='Z')
        # X logical: all data qubits (minimum weight transversal X)
        self.create_stim_logical({(float(2 * i), 0.0): 'X' for i in range(d)}, op_type='X')
        self.num_logicals = 1


# ── 2. Syndrome Extraction Block ─────────────────────────────────────────────

class BitFlipStripExtractionBlock:
    """One SE round: Reset syndrome → CX(data→syn) for left/right neighbors → Measure."""

    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self._build()

    def _build(self):
        c = self.circuit
        syn_indices = sorted(self.system.active_syndrome_indices_z)

        # Reset Z-ancillas
        c.append("R", syn_indices)
        # SE_start tick — noise injector attaches idle errors here
        c.append("TICK", tag="SE_start")

        # Two CX layers: left neighbor, then right neighbor
        # Use system.active_stabilizers_z + system.index_map for global indices
        for dx in [-1, +1]:
            pairs = []
            for stab in self.system.active_stabilizers_z:
                syn_coord = stab['syn_coord']
                neighbor = (syn_coord[0] + dx, syn_coord[1])
                if neighbor in self.system.index_map:
                    data_idx = self.system.index_map[neighbor]
                    pairs += [data_idx, stab['syn_idx']]  # CX: data→ancilla (Z-type)
            if pairs:
                c.append("CX", pairs)
            c.append("TICK")

        # Measure — MUST be the last instruction
        # CircuitBuilder reads this to determine measurement basis and qubit indices
        c.append("M", syn_indices)


# ── 3. Verify end-to-end ──────────────────────────────────────────────────────

def main():
    D = 5

    # Build code
    patch = BitFlipStrip(distance=D)
    print(f"BitFlipStrip d={D}: {patch.num_qubits} qubits, "
          f"{len(patch.stabilizers)} Z-stabilizers, {patch.num_logicals} logical")

    # Register in system
    system = QECSystem()
    system.add_patch(patch, name="strip")

    # Build circuit directly (not via MemoryExperiment)
    tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system)
    se = BitFlipStripExtractionBlock(system)

    builder.write_coordinates()
    builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
    builder.apply_syndrome_extraction(se.circuit, rounds=D)
    builder.apply_data_readout({q: "Z" for q in system.data_indices})

    # Noiseless sanity check — MUST pass before adding noise
    dets, obs = builder.circuit.compile_detector_sampler().sample(100, separate_observables=True)
    assert not dets.any(), "Noiseless circuit fires detectors — stabilizer or SE bug"
    assert not obs.any(),  "Noiseless circuit flips observable — logical operator bug"
    print(f"Noiseless check: PASS ({builder.circuit.num_detectors} detectors, "
          f"{builder.circuit.num_observables} observable)")

    # Add noise and simulate
    noisy = builder.build_noisy_circuit(
        NoiseConfig(p_2q=1e-3, p_meas=1e-3),
        noise_model="circuit_level",
    )
    stats = SimulationPipeline(
        decoder_config=DecoderConfig("pymatching"),
        max_errors=100, print_progress=False,
    ).run(noisy)
    print(f"LER = {stats.logical_error_rate:.3e} (p=1e-3, d={D})")

    print("\nTo add to the library:")
    print("  lightstim/qec_code/bit_flip_strip/code_patch.py  ← BitFlipStrip")
    print("  lightstim/qec_code/bit_flip_strip/SE_block.py    ← BitFlipStripExtractionBlock")
    print("  lightstim/qec_code/bit_flip_strip/__init__.py    ← export both classes")


if __name__ == "__main__":
    main()
