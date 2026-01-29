import stim

# -----------------------------------------------------------------------------
# Syndrome Extraction Block
# -----------------------------------------------------------------------------
class RepetitionCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit block for Repetition Code.
    
    This block represents ONE cycle of stabilizer measurements (Z-check):
    1. Reset syndrome qubits.
    2. CNOT Layer 1: Data(Left) -> Syndrome.
    3. CNOT Layer 2: Data(Right) -> Syndrome.
    4. Measure syndrome qubits.
    
    It relies on the 'system' object to provide coordinate-to-index mappings.
    NO NOISE is injected here; it is handled by an external noise_injector.
    """

    def __init__(self, system):
        """
        Args:
            system: The System object containing layout and index maps.
                    Expected attributes:
                    - index_map: Dict[(row, col), int]
                    - data_coords: List[(row, col)]
                    - syndrome_coords: List[(row, col)]
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
        
        # Critical Tag for NoiseInjector
        # Injectors (like CodeCapacity) look for this to inject errors on data qubits
        self.circuit.append("TICK", tag="SE_start") 

        # --- Step 2: CNOT Layers ---
        canonical_tick_deltas = [
            (-1, 0),  # Tick 1: Z checks Left
            (+1, 0)   # Tick 2: Z checks Right
        ]

        active_stabilizers_z = self.system.active_stabilizers_z

        for dx_z in canonical_tick_deltas:
            cnot_targets = []
            
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
                        # Data -> Syndrome (CNOT)
                        cnot_targets.extend([neighbor_idx, syn_idx])
            
            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            
            self.circuit.append("TICK")

        # --- Step 3: Measurement ---
        # Measure all syndrome qubits in Z basis
        self.circuit.append("M", active_syn_indices)