import stim


class XZZXSurfaceCodeExtractionBlock:
    """One syndrome-extraction cycle for the rotated XZZX surface code.

    Every ancilla is prepared and measured in the X basis (Hadamard before and
    after).  For each (syndrome, data) neighbour the entangling gate is chosen
    from the stabilizer's Pauli on that data qubit::

        'X'  ->  CNOT(syn, data)     # ancilla is control
        'Z'  ->  CZ(syn, data)       # symmetric

    The interaction schedule routes the two syndrome sublattices so that hook
    errors stay benign (matches the reference XZZX circuit): the SW and NE
    (X-diagonal) neighbours are coupled at ticks 0 and 3 (CNOT), while the NW and
    SE (Z-diagonal) neighbours are coupled at ticks 1 and 2 (CZ).

    Same constructor contract as ``RotatedSurfaceCodeExtractionBlock``: takes a
    ``system`` and exposes the built unit-cell circuit on ``.circuit``.
    """

    # Per tick: (offset for X-sublattice syndromes, offset for Z-sublattice).
    SCHEDULES = {
        "xzzx": [
            ((-1, -1), (-1, -1)),  # tick 0: both -> SW   (X-diagonal -> CNOT)
            ((+1, -1), (-1, +1)),  # tick 1: X->SE, Z->NW (Z-diagonal -> CZ)
            ((-1, +1), (+1, -1)),  # tick 2: X->NW, Z->SE (Z-diagonal -> CZ)
            ((+1, +1), (+1, +1)),  # tick 3: both -> NE   (X-diagonal -> CNOT)
        ],
    }

    def __init__(self, system, scheduling="xzzx"):
        self.system = system
        self.scheduling = scheduling
        if scheduling not in self.SCHEDULES:
            raise ValueError(
                f"Unknown scheduling {scheduling!r}. "
                f"Available: {sorted(self.SCHEDULES)}"
            )
        self.circuit = stim.Circuit()
        self._build_circuit()

    def _build_circuit(self):
        syn_all = sorted(self.system.active_syndrome_indices)

        # --- Reset all ancillas ---
        self.circuit.append("R", syn_all)
        self.circuit.append("TICK", tag="SE_start")

        # --- Prepare every ancilla in the X basis (XZZX ancilla) ---
        self.circuit.append("H", syn_all)
        self.circuit.append("TICK")

        stabs_x = self.system.active_stabilizers_x
        stabs_z = self.system.active_stabilizers_z

        # --- Entangling layers ---
        for dx_x, dx_z in self.SCHEDULES[self.scheduling]:
            cnot_targets = []
            cz_targets = []
            for stabs, delta in ((stabs_x, dx_x), (stabs_z, dx_z)):
                for stab in stabs:
                    syn_coord = stab["syn_coord"]
                    syn_idx = stab["syn_idx"]
                    owner_patch = self.system.patches[
                        self.system.coord_to_owner_map[syn_coord]
                    ][0]
                    delta_global = owner_patch.transform_vector(delta)
                    raw_target = (
                        syn_coord[0] + delta_global[0],
                        syn_coord[1] + delta_global[1],
                    )
                    target_key = owner_patch.get_grid_key(raw_target)
                    if target_key not in self.system.grid_map:
                        continue
                    neighbor_idx = self.system.grid_map[target_key]
                    if neighbor_idx not in stab["data_indices"]:
                        continue
                    pauli = stab["pauli"].get(neighbor_idx)
                    if pauli == "X":
                        cnot_targets.extend([syn_idx, neighbor_idx])
                    elif pauli == "Z":
                        cz_targets.extend([syn_idx, neighbor_idx])

            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            if cz_targets:
                self.circuit.append("CZ", cz_targets)
            self.circuit.append("TICK")

        # --- Basis change back + measure ---
        self.circuit.append("H", syn_all)
        self.circuit.append("TICK")
        self.circuit.append("M", syn_all)
