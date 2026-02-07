import stim

class ToricCodeExtractionBlock:
    """
    Syndrome extraction block for ToricCode using QECPatch stabilizers.

    Uses:
      - stab['syn_idx']
      - stab['data_indices']
      - stab['type'] ∈ {'X', 'Z'}
    """

    def __init__(self, patch):
        self.system = patch
        self.circuit = stim.Circuit()
        self._build_circuit()

    def _build_circuit(self):
        print("DEBUG: number of stabilizers seen by extraction block:", len(self.system.stabilizers))
        for i, s in enumerate(self.system.stabilizers):
            print(
                f"stab {i}: type={s['type']}, "
                f"syn_idx={s['syn_idx']}, "
                f"data_indices={s['data_indices']}"
            )


        # ------------------------------------------------------------
        # 1. Reset syndrome qubits
        # ------------------------------------------------------------
        syn_indices = [s["syn_idx"] for s in self.system.stabilizers]
        self.circuit.append("R", syn_indices)
        self.circuit.append("TICK", tag="SE_start")

        # ------------------------------------------------------------
        # 2. Prepare X-syndromes in |+>
        # ------------------------------------------------------------
        x_syn_indices = [s["syn_idx"] for s in self.system.stabilizers if s["type"] == "X"]
        if x_syn_indices:
            self.circuit.append("H", x_syn_indices)
        self.circuit.append("TICK")

        # ------------------------------------------------------------
        # 3. CNOT layers (6-tick unrotated schedule – toric code)
        # ------------------------------------------------------------

        canonical_tick_deltas = [
            ((0, 0), (-1, 0)),   # Tick 1
            ((0, 0), (+1, 0)),   # Tick 2
            ((0, +1), (0, +1)),  # Tick 3
            ((0, -1), (0, -1)),  # Tick 4
            ((-1, 0), (0, 0)),   # Tick 5
            ((+1, 0), (0, 0))    # Tick 6
        ]

        active_stabilizers_x = self.system.active_stabilizers_x
        active_stabilizers_z = self.system.active_stabilizers_z

        for dx_x, dx_z in canonical_tick_deltas:
            cnot_targets = []

            # -----------------------------
            # X stabilizers: ancilla → data
            # -----------------------------
            if dx_x != (0, 0):
                for stab in active_stabilizers_x:
                    syn_coord = stab["syn_coord"]
                    syn_idx = stab["syn_idx"]

                    owner_patch = self.system.patches[self.system.coord_to_owner_map[syn_coord]][0]

                    dx_global = owner_patch.transform_vector(dx_x)

                    # Compute toric wrap bounds
                    xs = [c[0] for c in owner_patch.qubit_coords.values()]
                    ys = [c[1] for c in owner_patch.qubit_coords.values()]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    width = max_x - min_x + 1
                    height = max_y - min_y + 1

                    # Wrap-around target
                    raw_target = (
                        min_x + (syn_coord[0] + dx_global[0] - min_x) % width,
                        min_y + (syn_coord[1] + dx_global[1] - min_y) % height,
                    )

                    target_key = owner_patch.get_grid_key(raw_target)
                    if target_key in owner_patch.grid_map:
                        data_idx = owner_patch.grid_map[target_key]
                        if data_idx in stab["data_indices"]:
                            cnot_targets.extend([syn_idx, data_idx])

            # -----------------------------
            # Z stabilizers: data → ancilla
            # -----------------------------
            if dx_z != (0, 0):
                for stab in active_stabilizers_z:
                    syn_coord = stab["syn_coord"]
                    syn_idx = stab["syn_idx"]

                    owner_patch = self.system.patches[self.system.coord_to_owner_map[syn_coord]][0]

                    dx_global = owner_patch.transform_vector(dx_z)

                    # Compute toric wrap bounds
                    xs = [c[0] for c in owner_patch.qubit_coords.values()]
                    ys = [c[1] for c in owner_patch.qubit_coords.values()]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    width = max_x - min_x + 1
                    height = max_y - min_y + 1

                    # Wrap-around target
                    raw_target = (
                        min_x + (syn_coord[0] + dx_global[0] - min_x) % width,
                        min_y + (syn_coord[1] + dx_global[1] - min_y) % height,
                    )

                    target_key = owner_patch.get_grid_key(raw_target)
                    if target_key in owner_patch.grid_map:
                        data_idx = owner_patch.grid_map[target_key]
                        if data_idx in stab["data_indices"]:
                            cnot_targets.extend([data_idx, syn_idx])

            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
                print("CNOT targets: ", cnot_targets)


            self.circuit.append("TICK")

        # ------------------------------------------------------------
        # 4. Rotate X-syndromes back
        # ------------------------------------------------------------
        if x_syn_indices:
            self.circuit.append("H", x_syn_indices)
        self.circuit.append("TICK")

        # ------------------------------------------------------------
        # 5. Measure syndromes
        # ------------------------------------------------------------
        self.circuit.append("M", syn_indices)

        # ------------------------------------------------------------
        # Debug print
        # ------------------------------------------------------------
        print("=== DEBUG: ToricCodeExtractionBlock ===")
        for tick, instr in enumerate(self.circuit):
            if instr.name in {"R", "H", "CX", "M"}:
                print(f"Tick {tick}: {instr}")
        print("======================================\n")
        print(self.circuit.diagram())
