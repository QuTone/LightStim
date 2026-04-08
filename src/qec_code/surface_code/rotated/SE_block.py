import stim

# -----------------------------------------------------------------------------
# Part 2. Syndrome Extraction Block
# -----------------------------------------------------------------------------
class RotatedSurfaceCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit block for Rotated Surface Code.

    This block represents ONE cycle of stabilizer measurements:
    1. Reset syndrome qubits.
    2. Entangling gates (H, CNOTs) following a configurable scheduling pattern.
    3. Measure syndrome qubits.

    It relies on the 'system' object to provide coordinate-to-index mappings.
    NO NOISE is injected here; it is handled by an external noise_injector.

    Args:
        system: QECSystem with rotated surface code patch(es).
        scheduling: CNOT scheduling variant.
            'perpendicular' (default) — fault-tolerant zigzag, achieves full code distance.
            'parallel' — non-FT zigzag, effective distance is halved due to hook errors.
    """

    SCHEDULES = {
        'perpendicular': [
            ((+1, +1), (+1, +1)),  # Tick 1: NE interaction
            ((-1, +1), (+1, -1)),  # Tick 2: NW / SE interaction (X and Z go different directions)
            ((+1, -1), (-1, +1)),  # Tick 3: SE / NW interaction (X and Z go different directions)
            ((-1, -1), (-1, -1)),  # Tick 4: SW interaction
        ],
        'parallel': [
            ((+1, +1), (+1, +1)),  # Tick 1: NE interaction
            ((-1, +1), (-1, +1)),  # Tick 2: NW interaction (X and Z go same direction → hook error)
            ((+1, -1), (+1, -1)),  # Tick 3: SE interaction (X and Z go same direction → hook error)
            ((-1, -1), (-1, -1)),  # Tick 4: SW interaction
        ],
        'swapped': [
            ((+1, +1), (+1, +1)),  # Tick 1: NE interaction
            ((+1, -1), (-1, +1)),  # Tick 2: X→SE, Z→NW  (swap of perpendicular tick 2/3)
            ((-1, +1), (+1, -1)),  # Tick 3: X→NW, Z→SE  (swap of perpendicular tick 2/3)
            ((-1, -1), (-1, -1)),  # Tick 4: SW interaction
        ],
    }

    def __init__(self, system, scheduling='perpendicular'):
        self.system = system
        self.scheduling = scheduling
        self.circuit = stim.Circuit()
        self._build_circuit()

    def _build_circuit(self):

        # --- Step 1: Reset Syndrome Qubits ---
        active_syn_indices = self.system.active_syndrome_indices
        self.circuit.append("R", sorted(active_syn_indices))
        self.circuit.append("TICK", tag="SE_start")

        # --- Step 2: Preparation (Hadamard on X-type syndromes) ---
        active_x_syn_indices = self.system.active_syndrome_indices_x
        self.circuit.append("H", sorted(active_x_syn_indices))
        self.circuit.append("TICK")

        # --- Step 3: Entangling Gates (CNOT Scheduling) ---
        canonical_tick_deltas = self.SCHEDULES[self.scheduling]

        # Get the active syndrome coordinates for X and Z stabilizers
        active_stabilizers_x = self.system.active_stabilizers_x
        active_stabilizers_z = self.system.active_stabilizers_z

        for dx_x, dx_z in canonical_tick_deltas:
            cnot_targets = []
            
            # 3.1 Handle X-Stabilizers (Syndrome is Control, Data is Target)
            for stab in active_stabilizers_x:
                syn_coord = stab['syn_coord']
                syn_idx = stab['syn_idx']
                owner_patch = self.system.patches[self.system.coord_to_owner_map[syn_coord]][0]
                dx_x_global = owner_patch.transform_vector(dx_x)
                raw_target = (
                    syn_coord[0] + dx_x_global[0], 
                    syn_coord[1] + dx_x_global[1]
                )
                target_key = owner_patch.get_grid_key(raw_target)

                if target_key in self.system.grid_map:
                    neighbor_idx = self.system.grid_map[target_key]
                    if neighbor_idx in stab['data_indices']:
                        cnot_targets.extend([syn_idx, neighbor_idx]) # Syndrome -> Data

            # 3.2 Handle Z-Stabilizers (Data is Control, Syndrome is Target)
            for stab in active_stabilizers_z:
                syn_coord = stab['syn_coord']
                syn_idx = stab['syn_idx']
                owner_patch = self.system.patches[self.system.coord_to_owner_map[syn_coord]][0]
                dx_z_global = owner_patch.transform_vector(dx_z)
                raw_target = (
                    syn_coord[0] + dx_z_global[0], 
                    syn_coord[1] + dx_z_global[1]
                )
                target_key = owner_patch.get_grid_key(raw_target)

                if target_key in self.system.grid_map:
                    neighbor_idx = self.system.grid_map[target_key]
                    if neighbor_idx in stab['data_indices']:
                        cnot_targets.extend([neighbor_idx, syn_idx]) # Data -> Syndrome

            # Apply CNOTs if any pairs exist in this tick
            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            
            self.circuit.append("TICK")

        # --- Step 4: Basis Change (Hadamard on X-type syndromes) ---
        # Transform X-syndrome qubits back to Z basis for measurement
        self.circuit.append("H", sorted(active_x_syn_indices))
        self.circuit.append("TICK")

        # --- Step 5: Measurement ---
        # Measure all syndrome qubits in Z basis
        self.circuit.append("M", sorted(active_syn_indices))
        
        # Note: No final TICK here. CircuitBuilder controls the flow.