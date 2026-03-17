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

        active_syn_indices = self.system.active_syndrome_indices
        active_x_syn_indices = self.system.active_syndrome_indices_x

        # Perpendicular zigzag scheduling (4 CNOT ticks)
        canonical_tick_deltas = [
            ((+1, +1), (+1, +1)), # Tick 1: NE
            ((-1, +1), (+1, -1)), # Tick 2: NW / SE
            ((+1, -1), (-1, +1)), # Tick 3: SE / NW
            ((-1, -1), (-1, -1)), # Tick 4: SW
        ]

        active_stabilizers_x = self.system.active_stabilizers_x
        active_stabilizers_z = self.system.active_stabilizers_z

        def _cnot_layer(dx_x, dx_z):
            """Build the CNOT target list for one tick."""
            targets = []
            for stab in active_stabilizers_x:
                syn_coord = stab['syn_coord']
                syn_idx   = stab['syn_idx']
                owner     = self.system.patches[self.system.coord_to_owner_map[syn_coord]][0]
                delta     = owner.transform_vector(dx_x)
                raw       = (syn_coord[0] + delta[0], syn_coord[1] + delta[1])
                key       = owner.get_grid_key(raw)
                if key in self.system.grid_map:
                    nb = self.system.grid_map[key]
                    if nb in stab['data_indices']:
                        targets.extend([syn_idx, nb])
            for stab in active_stabilizers_z:
                syn_coord = stab['syn_coord']
                syn_idx   = stab['syn_idx']
                owner     = self.system.patches[self.system.coord_to_owner_map[syn_coord]][0]
                delta     = owner.transform_vector(dx_z)
                raw       = (syn_coord[0] + delta[0], syn_coord[1] + delta[1])
                key       = owner.get_grid_key(raw)
                if key in self.system.grid_map:
                    nb = self.system.grid_map[key]
                    if nb in stab['data_indices']:
                        targets.extend([nb, syn_idx])
            return targets

        # ------------------------------------------------------------------
        # First half: Reset + H_x + CNOT ticks 1-2
        # ------------------------------------------------------------------
        first = stim.Circuit()
        first.append("R",    sorted(active_syn_indices))
        first.append("TICK", tag="SE_start")
        first.append("H",    sorted(active_x_syn_indices))
        first.append("TICK")
        for dx_x, dx_z in canonical_tick_deltas[:2]:
            targets = _cnot_layer(dx_x, dx_z)
            if targets:
                first.append("CNOT", targets)
            first.append("TICK")

        # ------------------------------------------------------------------
        # Second half: CNOT ticks 3-4 + H_x + Measure
        # ------------------------------------------------------------------
        second = stim.Circuit()
        for dx_x, dx_z in canonical_tick_deltas[2:]:
            targets = _cnot_layer(dx_x, dx_z)
            if targets:
                second.append("CNOT", targets)
            second.append("TICK")
        second.append("H", sorted(active_x_syn_indices))
        second.append("TICK")
        second.append("M", sorted(active_syn_indices))

        self.first_half = first
        self.second_half = second
        self.circuit = first + second
        # Note: No final TICK here. CircuitBuilder controls the flow.