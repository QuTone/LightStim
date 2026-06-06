"""Syndrome extraction block for Toric Code with periodic boundaries."""
import stim


class ToricCodeExtractionBlock:
    """
    Syndrome extraction block for Toric Code (unrotated with periodic boundaries).

    Uses the same CNOT schedules as UnrotatedSurfaceCodeExtractionBlock, with
    periodic wrap when resolving neighbor coordinates.

    Args:
        system: QECSystem with a toric code patch.
        scheduling: CNOT scheduling variant; one of SCHEDULES.
            '6tick' (default) — Li's-paper schedule that separates some X and Z
                layers to avoid conflicts on shared data qubits.
            '4tick' — minimal-depth schedule(perpendicular).

    Each entry is ``(dx_x, dx_z)``: the first element is the X-stabilizer offset
    (ancilla -> data), the second is the Z-stabilizer offset (data -> ancilla).
    ``(0, 0)`` means that stabilizer type does nothing on that tick.

    Expects system (QECSystem) with:
    - active_stabilizers_x, active_stabilizers_z
    - patches, coord_to_owner_map
    - grid_map (global coord -> index)
    - Stabilizer records with syn_coord, syn_idx, data_indices, type
    """

    # Own copy of the schedules (identical deltas to the unrotated block).
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
        if scheduling not in self.SCHEDULES:
            raise ValueError(
                f"Unknown scheduling {scheduling!r}. "
                f"Valid options: {sorted(self.SCHEDULES)}"
            )
        self.system = system
        self.scheduling = scheduling
        self.circuit = stim.Circuit()
        self._build_circuit()

    def _wrap_coord(self, patch, x: float, y: float) -> tuple:
        """Wrap (x,y) into patch bounds for periodic boundary conditions."""
        xs = [c[0] for c in patch.qubit_coords.values()]
        ys = [c[1] for c in patch.qubit_coords.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        wx = min_x + ((x - min_x) % width)
        wy = min_y + ((y - min_y) % height)
        return patch.snap_coord((wx, wy))

    def _build_circuit(self):
        active_syn_indices = self.system.active_syndrome_indices
        self.circuit.append("R", sorted(active_syn_indices))
        self.circuit.append("TICK", tag="SE_start")

        active_x_syn_indices = self.system.active_syndrome_indices_x
        self.circuit.append("H", sorted(active_x_syn_indices))
        self.circuit.append("TICK")

        canonical_tick_deltas = self.SCHEDULES[self.scheduling]

        active_stabilizers_x = self.system.active_stabilizers_x
        active_stabilizers_z = self.system.active_stabilizers_z

        for dx_x, dx_z in canonical_tick_deltas:
            cnot_targets = []

            if dx_x != (0, 0):
                for stab in active_stabilizers_x:
                    syn_coord = stab["syn_coord"]
                    syn_idx = stab["syn_idx"]
                    owner_patch = self.system.patches[
                        self.system.coord_to_owner_map[syn_coord]
                    ][0]
                    dx_global = owner_patch.transform_vector(dx_x)
                    raw_target = (
                        syn_coord[0] + dx_global[0],
                        syn_coord[1] + dx_global[1],
                    )
                    wrapped_target = self._wrap_coord(owner_patch, raw_target[0], raw_target[1])
                    target_key = owner_patch.get_grid_key(wrapped_target)
                    if target_key in self.system.grid_map:
                        neighbor_idx = self.system.grid_map[target_key]
                        if neighbor_idx in stab["data_indices"]:
                            cnot_targets.extend([syn_idx, neighbor_idx])

            if dx_z != (0, 0):
                for stab in active_stabilizers_z:
                    syn_coord = stab["syn_coord"]
                    syn_idx = stab["syn_idx"]
                    owner_patch = self.system.patches[
                        self.system.coord_to_owner_map[syn_coord]
                    ][0]
                    dx_global = owner_patch.transform_vector(dx_z)
                    raw_target = (
                        syn_coord[0] + dx_global[0],
                        syn_coord[1] + dx_global[1],
                    )
                    wrapped_target = self._wrap_coord(owner_patch, raw_target[0], raw_target[1])
                    target_key = owner_patch.get_grid_key(wrapped_target)
                    if target_key in self.system.grid_map:
                        neighbor_idx = self.system.grid_map[target_key]
                        if neighbor_idx in stab["data_indices"]:
                            cnot_targets.extend([neighbor_idx, syn_idx])

            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            self.circuit.append("TICK")

        self.circuit.append("H", sorted(active_x_syn_indices))
        self.circuit.append("TICK")

        self.circuit.append("M", sorted(active_syn_indices))
