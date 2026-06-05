import stim

# -----------------------------------------------------------------------------
# Part 2. Syndrome Extraction Block
# -----------------------------------------------------------------------------
class UnrotatedSurfaceCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit block for Unrotated Surface Code.

    This block represents ONE cycle of stabilizer measurements:
    1. Reset syndrome qubits.
    2. Entangling gates (H, CNOTs) following the selected CNOT scheduling.
    3. Measure syndrome qubits.

    It relies on the 'system' object to provide coordinate-to-index mappings.
    NO NOISE is injected here; it is handled by an external noise_injector.

    Args:
        system: An unrotated surface code patch.
        scheduling: CNOT scheduling variant (see SCHEDULES).
            '6tick' (default) — Li's-paper schedule that separates some X and Z
                layers to avoid conflicts on shared data qubits.
            '4tick' — minimal-depth schedule(perpendicular).

    Each entry is ``(dx_x, dx_z)``: the first element is the X-stabilizer offset
    (ancilla -> data, syndrome is control), the second is the Z-stabilizer offset
    (data -> ancilla, syndrome is target). ``(0, 0)`` means that stabilizer type
    does nothing on that tick.

    """

    SCHEDULES = {
        '6tick': [                    # Li's paper — DEFAULT
            ((0, 0), (-1, 0)),   # Tick 1
            ((0, 0), (+1, 0)),   # Tick 2
            ((0, +1), (0, +1)),  # Tick 3
            ((0, -1), (0, -1)),  # Tick 4
            ((-1, 0), (0, 0)),   # Tick 5
            ((+1, 0), (0, 0)),   # Tick 6
        ],
        '4tick': [                    # minimal-depth; X and Z mirror on the vertical ticks
            ((+1, 0), (+1, 0)),  # Tick 1: both East
            ((0, +1), (0, -1)),  # Tick 2: X North, Z South
            ((0, -1), (0, +1)),  # Tick 3: X South, Z North
            ((-1, 0), (-1, 0)),  # Tick 4: both West
        ],
    }

    def __init__(self, system, scheduling='6tick'):
        """
        Args:
            system: An unrotated surface code patch.
            scheduling: CNOT scheduling variant; one of SCHEDULES ('6tick', '4tick').
        """
        if scheduling not in self.SCHEDULES:
            raise ValueError(
                f"Unknown scheduling {scheduling!r}. "
                f"Valid options: {sorted(self.SCHEDULES)}"
            )
        self.system = system
        self.scheduling = scheduling
        self.circuit = stim.Circuit()

        # Build the circuit immediately upon instantiation
        self._build_circuit()

    def _build_circuit(self):

        # --- Step 1: Reset Syndrome Qubits ---
        # Reset all syndrome qubits to |0> (Z basis)
        active_syn_indices = self.system.active_syndrome_indices
        self.circuit.append("R", sorted(active_syn_indices))
        self.circuit.append("TICK", tag="SE_start") # NoiseInjector targets this tag

        # --- Step 2: Preparation (Hadamard on X-type syndromes) ---
        # Transform X-syndrome qubits to |+> state
        active_x_syn_indices = self.system.active_syndrome_indices_x
        self.circuit.append("H", sorted(active_x_syn_indices))
        self.circuit.append("TICK")

        # --- Step 3: Entangling Gates (CNOT Scheduling) ---
        # Format: (dx_x, dx_z)
        # dx_x: Offset for X-stabilizers (Ancilla -> Data)
        # dx_z: Offset for Z-stabilizers (Data -> Ancilla)
        canonical_tick_deltas = self.SCHEDULES[self.scheduling]

        # Get the active syndrome coordinates for X and Z stabilizers
        active_stabilizers_x = self.system.active_stabilizers_x
        active_stabilizers_z = self.system.active_stabilizers_z

        for dx_x, dx_z in canonical_tick_deltas:
            cnot_targets = []
            
            # 3.1 Handle X-Stabilizers (Syndrome is Control, Data is Target)
            # Only process if dx_x is not (0,0)
            if dx_x != (0, 0):
                # Go through all active X stabilizers
                for stab in active_stabilizers_x:
                    syn_coord = stab['syn_coord']
                    syn_idx = stab['syn_idx']
                    owner_patch = self.system.patches[self.system.coord_to_owner_map[syn_coord]][0]
                    dx_x_global = owner_patch.transform_vector(dx_x)
                    raw_target = (
                        syn_coord[0] + dx_x_global[0], 
                        syn_coord[1] + dx_x_global[1]
                    )
                    target_key = owner_patch.get_grid_key(raw_target) # get_grid_key is a static method and doesn't depend on the patch

                    if target_key in self.system.grid_map:
                        neighbor_idx = self.system.grid_map[target_key]
                        if neighbor_idx in stab['data_indices']:
                            cnot_targets.extend([syn_idx, neighbor_idx]) # Syndrome -> Data

            # 3.2 Handle Z-Stabilizers (Data is Control, Syndrome is Target)
            # Only process if dx_z is not (0,0)
            if dx_z != (0, 0):
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