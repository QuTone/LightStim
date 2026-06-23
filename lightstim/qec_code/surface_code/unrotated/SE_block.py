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
        active_stabilizers_mixed = getattr(self.system, 'active_stabilizers_mixed', [])

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

        # --- Step 3b: Mixed X/Z stabilizers ---
        # Mixed checks are measured from their explicit Pauli dictionaries.
        # Compatible checks are batched so routed/mixed surgery diagrams show
        # the ancillary patch region instead of a long sequence of single-check
        # fragments.  Mixed checks use an X-basis syndrome ancilla: Z terms
        # couple via CZ(data, syndrome), while X terms couple via
        # CNOT(syndrome, data).  Entangling edges are layer-colored to avoid
        # shared-qubit conflicts in the same tick.
        self._append_mixed_stabilizer_measurements(active_stabilizers_mixed)

        # --- Step 4: Basis Change (Hadamard on X-type syndromes) ---
        # Transform X-syndrome qubits back to Z basis for measurement
        self.circuit.append("H", sorted(active_x_syn_indices))
        self.circuit.append("TICK")

        # --- Step 5: Measurement ---
        # Measure all syndrome qubits in Z basis
        self.circuit.append("M", sorted(active_syn_indices))

    # Hook-benign directional schedule for MIXED (domain-wall) checks.
    # Each mixed check is measured with ONE X-basis ancilla: X terms couple via
    # CNOT(ancilla -> data), Z terms via CZ(ancilla, data).  The only undetected
    # single-fault hook is an X (or Y) error on the ancilla mid-sequence; by exact
    # tableau propagation it deposits X on every *subsequent* CNOT-data qubit and Z on
    # every *subsequent* CZ-data qubit.  Its geometry is therefore fixed entirely by the
    # gate ORDER.  We schedule each entangling gate in the SAME direction-by-tick slot
    # the corresponding pure check uses (SCHEDULES['6tick']), so the mixed hook inherits
    # the pure checks' benign (logical-perpendicular) orientation by construction instead
    # of the previous ad-hoc per-data-index order.  Each entry is
    # (x_term_direction, z_term_direction) as the (dx, dy) offset of the data qubit from
    # the ancilla; None means that Pauli type does nothing on that tick.
    MIXED_TICKS = [
        (None,    (-1, 0)),   # tick 1: Z term West
        (None,    (+1, 0)),   # tick 2: Z term East
        ((0, +1), (0, +1)),   # tick 3: X term North, Z term North
        ((0, -1), (0, -1)),   # tick 4: X term South, Z term South
        ((-1, 0), None),      # tick 5: X term West
        ((+1, 0), None),      # tick 6: X term East
    ]

    def _append_mixed_stabilizer_measurements(self, stabs):
        if not stabs:
            return

        coords = self.system.qubit_coords

        def direction(anc_coord, data_idx):
            ax, ay = anc_coord[0], anc_coord[1]
            dx, dy = coords[data_idx][0], coords[data_idx][1]
            return ((dx > ax) - (dx < ax), (dy > ay) - (dy < ay))

        # Checks are grouped into compatibility batches: within a batch every shared data
        # qubit carries the SAME Pauli, so all CNOT/CZ on a shared qubit commute and the
        # gates may be freely reordered.  (Across batches a shared qubit can have opposite
        # Paulis -- CNOT and CZ there do NOT commute -- so those checks must be measured in
        # separate H...H blocks; that is exactly what the batching guarantees.)  Within
        # each batch we apply the hook-benign directional schedule.
        for batch in self._batch_compatible_mixed_stabilizers(stabs):
            mixed_syn = sorted({stab['syn_idx'] for stab in batch})
            self.circuit.append("H", mixed_syn)
            self.circuit.append("TICK")

            emitted = set()
            for x_dir, z_dir in self.MIXED_TICKS:
                cnot_targets = []
                cz_targets = []
                for stab in batch:
                    syn_idx = stab['syn_idx']
                    syn_coord = stab['syn_coord']
                    for data_idx, pauli in stab['pauli'].items():
                        if (syn_idx, data_idx) in emitted:
                            continue
                        dvec = direction(syn_coord, data_idx)
                        if pauli == 'X' and x_dir is not None and dvec == x_dir:
                            cnot_targets.extend([syn_idx, data_idx])      # CNOT(anc -> data)
                            emitted.add((syn_idx, data_idx))
                        elif pauli == 'Z' and z_dir is not None and dvec == z_dir:
                            cz_targets.extend([data_idx, syn_idx])        # CZ(data, anc)
                            emitted.add((syn_idx, data_idx))
                # Batch-level empty-tick compression: emit the TICK only when this
                # direction slot carries at least one gate for SOME check in the batch.
                # A fully-empty column is pure idle; dropping it removes no gate and
                # changes no gate's relative order, so per-check hook structure and the
                # cross-check directional synchronization are preserved -- only idle
                # (and its idling noise) is removed.  (A column that is empty for only
                # some checks is kept: those checks simply idle that tick.)
                if cnot_targets or cz_targets:
                    if cnot_targets:
                        self.circuit.append("CNOT", cnot_targets)
                    if cz_targets:
                        self.circuit.append("CZ", cz_targets)
                    self.circuit.append("TICK")

            # Safety net: a term whose data qubit is not an axis-unit neighbour of the
            # ancilla (unusual geometry) matches no direction slot; emit it last so the
            # stabilizer is always measured in full.
            leftover_cnot = []
            leftover_cz = []
            for stab in batch:
                syn_idx = stab['syn_idx']
                for data_idx, pauli in stab['pauli'].items():
                    if (syn_idx, data_idx) in emitted:
                        continue
                    if pauli == 'X':
                        leftover_cnot.extend([syn_idx, data_idx])
                    else:
                        leftover_cz.extend([data_idx, syn_idx])
                    emitted.add((syn_idx, data_idx))
            if leftover_cnot or leftover_cz:
                if leftover_cnot:
                    self.circuit.append("CNOT", leftover_cnot)
                if leftover_cz:
                    self.circuit.append("CZ", leftover_cz)
                self.circuit.append("TICK")

            self.circuit.append("H", mixed_syn)
            self.circuit.append("TICK")

    @staticmethod
    def _batch_compatible_mixed_stabilizers(stabs):
        """Group mixed checks so that within a batch every shared data qubit carries the
        same Pauli.  This keeps all intra-batch CNOT/CZ on shared qubits mutually
        commuting (so the directional reorder is safe) and forces checks that disagree on
        a shared qubit into separate H...H measurement blocks."""
        batches = []
        batch_paulis = []

        for stab in stabs:
            paulis = stab.get('pauli', {})
            for pauli in paulis.values():
                if pauli not in ('X', 'Z'):
                    raise ValueError(f"Mixed stabilizer only supports X/Z terms, got {pauli!r}.")

            placed = False
            for batch, seen in zip(batches, batch_paulis):
                if all(seen.get(q, pauli) == pauli for q, pauli in paulis.items()):
                    batch.append(stab)
                    for q, pauli in paulis.items():
                        seen[q] = pauli
                    placed = True
                    break

            if not placed:
                batches.append([stab])
                batch_paulis.append(dict(paulis))

        return batches
