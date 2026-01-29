import stim

# -----------------------------------------------------------------------------
# Part 2. Syndrome Extraction Block
# -----------------------------------------------------------------------------
class RotatedSurfaceCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit block for Rotated Surface Code.
    
    This block represents ONE cycle of stabilizer measurements:
    1. Reset syndrome qubits.
    2. Entangling gates (H, CNOTs) following the "Z" (or "N") scheduling pattern.
    3. Measure syndrome qubits.
    
    It relies on the 'system' object to provide coordinate-to-index mappings.
    NO NOISE is injected here; it is handled by an external noise_injector.
    """

    def __init__(self, system):
        """
        Args:
            system: A rotated surface code patch.
        """
        self.system = system
        self.circuit = stim.Circuit()
        
        # Build the circuit immediately upon instantiation
        self._build_circuit()

    def _build_circuit(self):

        # --- Step 1: Reset Syndrome Qubits ---
        # Reset all syndrome qubits to |0> (Z basis)
        active_syn_indices = self.system.active_syndrome_indices
        self.circuit.append("R", active_syn_indices)
        self.circuit.append("TICK", tag="SE_start") # DEPOLARIZE1 will be injected here on data qubits

        # --- Step 2: Preparation (Hadamard on X-type syndromes) ---
        # Transform X-syndrome qubits to |+> state to measure X operators
        active_x_syn_indices = self.system.active_syndrome_indices_x
        self.circuit.append("H", active_x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 3: Entangling Gates (CNOT Scheduling) ---
        # Standard Rotated Surface Code scheduling (4 ticks)
        # Format: ((dx_x, dy_x), (dx_z, dy_z))
        # dx_x/dy_x is the offset for X-stabilizer checks
        # dx_z/dy_z is the offset for Z-stabilizer checks
        # Perpendicular zigzag scheduling
        canonical_tick_deltas = [
            ((+1, +1), (+1, +1)), # Tick 1: NE interaction
            ((-1, +1), (+1, -1)), # Tick 2: NW / SE interaction
            ((+1, -1), (-1, +1)), # Tick 3: SE / NW interaction
            ((-1, -1), (-1, -1))  # Tick 4: SW interaction
        ]

        # Parallel zigzag scheduling (not FT, half distsance)
        # canonical_tick_deltas = [
        #     ((+1, +1), (+1, +1)), # Tick 1: NE interaction
        #     ((-1, +1), (-1, +1)), # Tick 2: NW / SE interaction
        #     ((+1, -1), (+1, -1)), # Tick 3: SE / NW interaction
        #     ((-1, -1), (-1, -1))  # Tick 4: SW interaction
        # ]

        # Get the active syndrome coordinates for X and Z stabilizers
        active_stabilizers_x = self.system.active_stabilizers_x
        active_stabilizers_z = self.system.active_stabilizers_z

        for dx_x, dx_z in canonical_tick_deltas:
            cnot_targets = []
            
            # 3.1 Handle X-Stabilizers (Syndrome is Control, Data is Target)
            for stab in active_stabilizers_x:
                syn_coord = stab['syn_coord']
                syn_idx = stab['syn_idx']
                owner_patch = self.system.patches[self.system.owner_map[syn_coord]][0]
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
                owner_patch = self.system.patches[self.system.owner_map[syn_coord]][0]
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
        self.circuit.append("H", active_x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 5: Measurement ---
        # Measure all syndrome qubits in Z basis
        self.circuit.append("M", active_syn_indices)
        
        # Note: No final TICK here. CircuitBuilder controls the flow.