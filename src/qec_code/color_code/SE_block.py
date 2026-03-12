"""Syndrome extraction block for the Triangular Color Code.

Implements space-multiplexed syndrome extraction with a 7-timeslice
CNOT schedule (optimal from Lee et al., color-code-stim).

Each hexagonal face has:
- Z-ancilla: data qubits control ancilla (CNOT data→anc)
- X-ancilla: ancilla controls data qubits (CNOT anc→data)
"""

import stim


class ColorCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit for the Color Code.

    Uses a 7-timeslice CNOT schedule with 6 CNOT positions per stabilizer
    type (Z and X), for a total of 12 positions per face.

    Optimal schedule A = [2,3,6,5,4,1; 3,4,7,6,5,2] from Lee et al.
    """

    # 6 offsets from face center to data qubit vertices (same for X and Z).
    OFFSETS = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)]

    # 7-timeslice schedule.
    # Each entry: (z_position_index_or_None, x_position_index_or_None)
    # Position index into OFFSETS array.
    # Optimal schedule from Lee et al.: A = [2,3,6,5,4,1, 3,4,7,6,5,2]
    # Z-positions [0-5] → timeslices [2,3,6,5,4,1]
    # X-positions [6-11] → timeslices [3,4,7,6,5,2]
    SCHEDULE = [
        (5, None),   # Timeslice 1: Z-pos5
        (0, 5),      # Timeslice 2: Z-pos0, X-pos5
        (1, 0),      # Timeslice 3: Z-pos1, X-pos0
        (4, 1),      # Timeslice 4: Z-pos4, X-pos1
        (3, 4),      # Timeslice 5: Z-pos3, X-pos4
        (2, 3),      # Timeslice 6: Z-pos2, X-pos3
        (None, 2),   # Timeslice 7: X-pos2
    ]

    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self._extract_color_code_params()
        self._build_circuit()

    def _extract_color_code_params(self):
        """Extract color code parameters from the patch in the system."""
        for name, (patch, _) in self.system.patches.items():
            if hasattr(patch, 'faces') and hasattr(patch, 'FACE_OFFSETS'):
                self._patch = patch
                self._faces = patch.faces
                return
        raise ValueError("No ColorCode patch found in the system.")

    def _build_circuit(self):
        # --- Step 1: Reset syndrome qubits ---
        active_syn_indices = self.system.active_syndrome_indices
        self.circuit.append("R", active_syn_indices)
        self.circuit.append("TICK", tag="SE_start")

        # --- Step 2: Hadamard on X-type syndromes ---
        active_x_syn_indices = self.system.active_syndrome_indices_x
        self.circuit.append("H", active_x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 3: 7-timeslice CNOT schedule ---
        for z_pos, x_pos in self.SCHEDULE:
            cnot_targets = []

            # Z-stabilizer CNOTs (data → ancilla)
            if z_pos is not None:
                offset = self.OFFSETS[z_pos]
                for face in self._faces:
                    cx, cy = face['center']
                    data_coord = self._patch.snap_coord(
                        (cx + offset[0], cy + offset[1])
                    )
                    data_key = self._patch.get_grid_key(data_coord)
                    if data_key in self.system.grid_map:
                        data_idx = self.system.grid_map[data_key]
                        if data_idx in self.system.data_indices:
                            z_anc_idx = face['z_ancilla_idx']
                            cnot_targets.extend([data_idx, z_anc_idx])

            # X-stabilizer CNOTs (ancilla → data)
            if x_pos is not None:
                offset = self.OFFSETS[x_pos]
                for face in self._faces:
                    cx, cy = face['center']
                    data_coord = self._patch.snap_coord(
                        (cx + offset[0], cy + offset[1])
                    )
                    data_key = self._patch.get_grid_key(data_coord)
                    if data_key in self.system.grid_map:
                        data_idx = self.system.grid_map[data_key]
                        if data_idx in self.system.data_indices:
                            x_anc_idx = face['x_ancilla_idx']
                            cnot_targets.extend([x_anc_idx, data_idx])

            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            self.circuit.append("TICK")

        # --- Step 4: Hadamard on X-type syndromes (back to Z basis) ---
        self.circuit.append("H", active_x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 5: Measurement ---
        self.circuit.append("M", active_syn_indices)
