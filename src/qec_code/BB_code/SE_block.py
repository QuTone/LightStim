"""Syndrome extraction block for Bivariate Bicycle (BB) codes."""
import stim


class BBCodeExtractionBlock:
    """
    Generates the noiseless syndrome extraction circuit block for BB codes.

    One cycle of stabilizer measurements using a 7-tick CNOT schedule
    (matching Bravyi et al. 2024 and the original BBcode.py).

    The schedule interleaves X and Z stabilizer CNOTs:
    - X-stabs: syndrome (control) -> data (target)
    - Z-stabs: data (control) -> syndrome (target)

    NO NOISE is injected here; it is handled by NoiseInjector externally.
    """

    # 7-tick CNOT schedule.
    # Each entry: (z_action, x_action)
    # z_action/x_action = None means idle for that type
    # Otherwise: (matrix_name, row_index, negate, coord_type)
    #   matrix_name: 'A' or 'B'
    #   row_index: 0, 1, or 2
    #   negate: True if offset should be negated (for Z-stabs)
    #   coord_type: 'left' (even x, odd y) or 'right' (odd x, even y)
    SCHEDULE = [
        # Tick 1: Z uses -A[0] (right-type), X idle
        (('A', 0, True, 'right'), None),
        # Tick 2: Z uses -A[2] (right-type), X uses A[1] (left-type)
        (('A', 2, True, 'right'), ('A', 1, False, 'left')),
        # Tick 3: Z uses -B[0] (left-type), X uses B[1] (right-type)
        (('B', 0, True, 'left'), ('B', 1, False, 'right')),
        # Tick 4: Z uses -B[1] (left-type), X uses B[0] (right-type)
        (('B', 1, True, 'left'), ('B', 0, False, 'right')),
        # Tick 5: Z uses -B[2] (left-type), X uses B[2] (right-type)
        (('B', 2, True, 'left'), ('B', 2, False, 'right')),
        # Tick 6: Z uses -A[1] (right-type), X uses A[0] (left-type)
        (('A', 1, True, 'right'), ('A', 0, False, 'left')),
        # Tick 7: Z idle, X uses A[2] (left-type)
        (None, ('A', 2, False, 'left')),
    ]

    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self._extract_bb_params()
        self._build_circuit()

    def _extract_bb_params(self):
        """Extract BB code algebraic parameters from the patch."""
        # Find the BBCode patch in the system
        for name, (patch, _) in self.system.patches.items():
            if hasattr(patch, 'A') and hasattr(patch, 'B') and hasattr(patch, 'l'):
                self._patch = patch
                self._A = patch.A
                self._B = patch.B
                self._l = patch.l
                self._m = patch.m
                return
        raise ValueError("No BBCode patch found in the system.")

    def _wrap_coord(self, x: float, y: float) -> tuple:
        """Wrap (x, y) into the torus [0, 2l) x [0, 2m)."""
        l, m = self._l, self._m
        wx = x % (2 * l)
        wy = y % (2 * m)
        return self._patch.snap_coord((wx, wy))

    def _compute_data_coord(self, anc_x: int, anc_y: int,
                            matrix_name: str, row_idx: int,
                            negate: bool, coord_type: str) -> tuple:
        """
        Compute the target data qubit coordinate for a CNOT from a given
        syndrome ancilla, using the specified polynomial offset.

        Args:
            anc_x, anc_y: Integer coordinates of the ancilla qubit.
            matrix_name: 'A' or 'B'.
            row_idx: Row index (0, 1, 2) in the polynomial matrix.
            negate: Whether to negate the offset (for Z-stabilizers).
            coord_type: 'left' (even x, odd y) or 'right' (odd x, even y).
        """
        l, m = self._l, self._m
        poly = self._A if matrix_name == 'A' else self._B
        dx, dy = poly[row_idx]

        if negate:
            dx, dy = -dx, -dy

        # Ancilla grid position: j = anc_x // 2, i = anc_y // 2
        j = anc_x // 2
        i = anc_y // 2

        if coord_type == 'left':
            # Left-type data: even x, odd y → (2*col, 2*row + 1)
            data_x = 2 * ((j + dx) % l)
            data_y = 2 * ((i + dy) % m) + 1
        else:  # 'right'
            # Right-type data: odd x, even y → (2*col + 1, 2*row)
            data_x = 2 * ((j + dx) % l) + 1
            data_y = 2 * ((i + dy) % m)

        return self._wrap_coord(data_x, data_y)

    def _build_circuit(self):
        # --- Step 1: Reset syndrome qubits ---
        active_syn_indices = self.system.active_syndrome_indices
        self.circuit.append("R", active_syn_indices)
        self.circuit.append("TICK", tag="SE_start")

        # --- Step 2: Hadamard on X-type syndromes ---
        active_x_syn_indices = self.system.active_syndrome_indices_x
        self.circuit.append("H", active_x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 3: 7-tick CNOT schedule ---
        active_stabilizers_x = self.system.active_stabilizers_x
        active_stabilizers_z = self.system.active_stabilizers_z

        for z_action, x_action in self.SCHEDULE:
            cnot_targets = []

            # Z-stabilizer CNOTs (data -> syndrome)
            if z_action is not None:
                matrix_name, row_idx, negate, coord_type = z_action
                for stab in active_stabilizers_z:
                    syn_coord = stab['syn_coord']
                    syn_idx = stab['syn_idx']
                    anc_x = int(round(syn_coord[0]))
                    anc_y = int(round(syn_coord[1]))

                    data_coord = self._compute_data_coord(
                        anc_x, anc_y, matrix_name, row_idx, negate, coord_type
                    )
                    target_key = self._patch.get_grid_key(data_coord)
                    if target_key in self.system.grid_map:
                        data_idx = self.system.grid_map[target_key]
                        cnot_targets.extend([data_idx, syn_idx])  # data -> syndrome

            # X-stabilizer CNOTs (syndrome -> data)
            if x_action is not None:
                matrix_name, row_idx, negate, coord_type = x_action
                for stab in active_stabilizers_x:
                    syn_coord = stab['syn_coord']
                    syn_idx = stab['syn_idx']
                    anc_x = int(round(syn_coord[0]))
                    anc_y = int(round(syn_coord[1]))

                    data_coord = self._compute_data_coord(
                        anc_x, anc_y, matrix_name, row_idx, negate, coord_type
                    )
                    target_key = self._patch.get_grid_key(data_coord)
                    if target_key in self.system.grid_map:
                        data_idx = self.system.grid_map[target_key]
                        cnot_targets.extend([syn_idx, data_idx])  # syndrome -> data

            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            self.circuit.append("TICK")

        # --- Step 4: Hadamard on X-type syndromes (back to Z basis) ---
        self.circuit.append("H", active_x_syn_indices)
        self.circuit.append("TICK")

        # --- Step 5: Measurement ---
        self.circuit.append("M", active_syn_indices)
