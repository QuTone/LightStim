"""Syndrome extraction block for 4D geometric codes.

Implements the compact circuit (Circuit 2 from arXiv:2506.15130, Appendix A):
  - X and Z ancillas prepared and measured simultaneously
  - 8-tick CNOT schedule scanning directions: -3, -2, -1, -0, +0, +1, +2, +3
  - Total depth: 8 CNOT ticks + 2 H layers + reset/measure = depth 12

NO NOISE is injected here; it is handled by NoiseInjector externally.
"""

import stim
from .lattice import Lattice4D


class FourDGeoCodeExtractionBlock:
    """Compact syndrome extraction circuit for 4D geometric codes.

    Circuit structure:
        1. Reset all syndrome qubits
        2. Hadamard on X-type syndromes
        3. 8 CNOT ticks (one per signed direction)
           - X-stab: syndrome (control) → data (target)
           - Z-stab: data (control) → syndrome (target)
        4. Hadamard on X-type syndromes
        5. Measure all syndrome qubits
    """

    # Compact circuit direction order (Circuit 2, Appendix A)
    # Each entry: (sign, axis)
    DIRECTION_ORDER = [
        (-1, 3), (-1, 2), (-1, 1), (-1, 0),
        (+1, 0), (+1, 1), (+1, 2), (+1, 3),
    ]

    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self._extract_patch()
        self._build_circuit()

    def _extract_patch(self):
        """Find the FourDGeoCode patch in the system."""
        for name, (patch, _) in self.system.patches.items():
            if hasattr(patch, '_lattice') and hasattr(patch, '_face_qubit_map'):
                self._patch = patch
                self._lattice = patch._lattice
                return
        raise ValueError("No FourDGeoCode patch found in the system.")

    def _build_circuit(self):
        circuit = self.circuit
        patch = self._patch
        lattice = self._lattice

        # Get global syndrome indices from the system
        x_syn = sorted(self.system.active_syndrome_indices_x)
        z_syn = sorted(self.system.active_syndrome_indices_z)
        all_syn = sorted(set(x_syn + z_syn))

        # Build local-to-global maps for the patch's qubit maps
        # We need to map patch-local qubit indices to system-global indices
        local_to_global = {}
        for lname, mapping in self.system.local_to_global_map.items():
            local_to_global.update(mapping)

        # Build face/edge/cube lookup: (cell_key) → global_qubit_index
        face_to_global = {}
        for key, local_idx in patch._face_qubit_map.items():
            face_to_global[key] = local_to_global[local_idx]

        edge_to_global = {}
        for key, local_idx in patch._edge_qubit_map.items():
            edge_to_global[key] = local_to_global[local_idx]

        cube_to_global = {}
        for key, local_idx in patch._cube_qubit_map.items():
            cube_to_global[key] = local_to_global[local_idx]

        # 1. Reset all syndrome qubits
        circuit.append("R", all_syn)
        circuit.append("TICK")

        # 2. Hadamard on X-type syndromes
        circuit.append("H", x_syn)
        circuit.append("TICK")

        # 3. Eight CNOT ticks
        points = lattice.enumerate_points()

        for sign, axis in self.DIRECTION_ORDER:
            cnot_pairs = []

            # X-stabilizer CNOTs: syndrome (control) → data (target)
            for edir in range(4):
                for p in points:
                    face = lattice.se_edge_to_face(sign, axis, edir, p)
                    if face is not None:
                        syn_global = edge_to_global[(edir, p)]
                        data_global = face_to_global[face]
                        cnot_pairs.append((syn_global, data_global))

            # Z-stabilizer CNOTs: data (control) → syndrome (target)
            for cmiss in range(4):
                for p in points:
                    face = lattice.se_cube_to_face(sign, axis, cmiss, p)
                    if face is not None:
                        data_global = face_to_global[face]
                        syn_global = cube_to_global[(cmiss, p)]
                        cnot_pairs.append((data_global, syn_global))

            # Append all CNOTs for this tick
            cnot_targets = []
            for ctrl, tgt in cnot_pairs:
                cnot_targets.extend([ctrl, tgt])
            if cnot_targets:
                circuit.append("CNOT", cnot_targets)
            circuit.append("TICK")

        # 4. Hadamard back on X-type syndromes
        circuit.append("H", x_syn)
        circuit.append("TICK")

        # 5. Measure all syndrome qubits
        circuit.append("M", all_syn)
