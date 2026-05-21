"""
Template for adding a new QEC code to LightStim.

Every code requires two files:
  code_patch.py  — subclass QECPatch, define geometry + stabilizers + logicals
  SE_block.py    — class with a .circuit attribute, one round of syndrome extraction

This script shows a minimal runnable example: a single-row repetition-like code
called "BitFlipStrip" that can be dropped into lightstim/qec_code/your_code/.

Key QECPatch API used here:
  self.add_qubit(x, y, role)                  — register qubit, role ∈ {'data', 'syndrome_z', 'syndrome_x'}
  self.create_stim_stabilizer(target_dict, syn_coord, type)
      target_dict: Dict[(x, y) → 'X'|'Z'|'Y'],  type: 'X' or 'Z'
  self.create_stim_logical(target_dict, op_type)
      op_type: 'X' or 'Z' (the logical operator type)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import stim
from lightstim.ir.qec_patch import QECPatch
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.noise.config import NoiseConfig


# ── 1. Code Patch ────────────────────────────────────────────────────────────

class BitFlipStrip(QECPatch):
    """
    Distance-d repetition code (Z-stabilizers only, corrects bit-flip errors).

    Layout along x-axis:
      D  S  D  S  D  ...  D
      0  1  2  3  4       2d-2
    D = data (even indices), S = syndrome_z (odd indices)
    """

    def _process_params(self):
        self.distance = self.params.get('distance', 3)
        if self.distance < 2:
            raise ValueError("distance must be >= 2")

    def build(self):
        d = self.distance

        # Phase 1: geometry — register all qubits
        for x in range(2 * d - 1):
            role = 'data' if x % 2 == 0 else 'syndrome_z'
            self.add_qubit(x, 0, role=role)

        # Phase 2: physics — one ZZ stabilizer per syndrome qubit
        for syn_coord in self.syndrome_coords:
            left  = (syn_coord[0] - 1, 0)
            right = (syn_coord[0] + 1, 0)
            self.create_stim_stabilizer(
                {left: 'Z', right: 'Z'},
                syn_coord=syn_coord,
                type='Z',
            )

        # Phase 3: logical operators (both X and Z representative for each logical)
        self.create_stim_logical({(0, 0): 'Z'}, op_type='Z')
        self.create_stim_logical({(2 * i, 0): 'X' for i in range(d)}, op_type='X')
        self.num_logicals = 1


# ── 2. Syndrome Extraction Block ─────────────────────────────────────────────

class BitFlipStripExtractionBlock:
    """One round: Reset → CX(data→syn) × 2 → Measure."""

    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self._build()

    def _build(self):
        c = self.circuit
        syn_indices = sorted(self.system.syndrome_indices)
        c.append("R", syn_indices)
        c.append("TICK", tag="SE_start")

        for dx in [-1, +1]:   # left neighbor, right neighbor
            pairs = []
            for stab in self.system.active_stabilizers_z:
                syn_coord = stab['syn_coord']
                neighbor = (syn_coord[0] + dx, syn_coord[1])
                if neighbor in self.system.index_map:
                    pairs += [self.system.index_map[neighbor], stab['syn_idx']]
            if pairs:
                c.append("CX", pairs)
            c.append("TICK")

        c.append("M", syn_indices)


# ── 3. Verify it works ────────────────────────────────────────────────────────

def main():
    patch = BitFlipStrip(distance=5)
    print(f"BitFlipStrip d=5: {patch.num_qubits} qubits, "
          f"{len(patch.stabilizers)} stabilizers, "
          f"{patch.num_logicals} logical(s)")

    system = QECSystem()
    system.add_patch(patch, name='strip')

    noise = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3)
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=BitFlipStripExtractionBlock,
        rounds=5,
        noise_params=noise,
        noise_model='circuit_level',
        basis='Z',
    )
    circuit = exp.build()
    print(f"Memory circuit: {circuit.num_qubits} qubits, "
          f"{circuit.num_detectors} detectors, "
          f"{circuit.num_observables} observable(s)")

    # Noiseless sanity check
    noiseless = exp.builder.circuit
    sampler = noiseless.compile_detector_sampler()
    det_events, _ = sampler.sample(shots=20, separate_observables=True)
    assert det_events.sum() == 0, "Unexpected detection events"
    print("Noiseless check passed: 0 detection events over 20 shots")
    print("\nTo add this code to the library, move the two classes into:")
    print("  lightstim/qec_code/bit_flip_strip/code_patch.py")
    print("  lightstim/qec_code/bit_flip_strip/SE_block.py")


if __name__ == '__main__':
    main()
