"""Syndrome extraction blocks for color-code measurement layouts.

The :class:`ColorCodeSpaceMultiplexingBlock` is the original two-ancilla circuit
on the superdense layout. The remaining blocks reproduce the alternative
measurement schedules from Fig. 2 of the color-code paper.
"""

import stim

from lightstim.qec_code.color_code.midout_plan import RectangleMiddleOutPlan


class _ColorCodeBlockBase:
    """Minimal shared plumbing for color-code SE blocks."""

    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self._extract_color_code_params()
        self._build_circuit()
        if not hasattr(self, "measurement_blocks"):
            self.measurement_blocks = (self.circuit,)

    def _extract_color_code_params(self):
        for name, (patch, _) in self.system.patches.items():
            if (
                hasattr(patch, "faces")
                and hasattr(patch, "tiles")
                and getattr(patch, "layout", None) in getattr(patch, "SUPPORTED_LAYOUTS", ())
            ):
                self._patch = patch
                self._faces = patch.faces
                self.layout = patch.layout
                self._local_to_global = self.system.local_to_global_map.get(name, {})
                return
        raise ValueError("No ColorCode patch found in the system.")

    def _require_layout(self, *layouts):
        if self.layout not in layouts:
            allowed = ", ".join(repr(layout) for layout in layouts)
            raise ValueError(
                f"{self.__class__.__name__} requires color-code layout "
                f"{allowed}; got {self.layout!r}."
            )

    def _g(self, local_idx):
        return self._local_to_global.get(local_idx, local_idx)

    def _tile_neighbor(self, face, basis, pos):
        tile = face["x_tile"] if basis == "X" else face["z_tile"]
        if tile.ordered_data_coords[pos] is None:
            return None
        neighbor = face["data_neighbors"][pos]
        if neighbor is None:
            return None
        _, local_idx = neighbor
        data_idx = self._g(local_idx)
        return data_idx if data_idx in self.system.data_indices else None

    def _x_neighbor(self, face, pos):
        return self._tile_neighbor(face, "X", pos)

    def _z_neighbor(self, face, pos):
        return self._tile_neighbor(face, "Z", pos)

    def _neighbor(self, face, pos):
        return self._tile_neighbor(face, "X", pos)

    def _z_ancilla(self, face):
        idx = face.get("z_ancilla_idx")
        return None if idx is None else self._g(idx)

    def _x_ancilla(self, face):
        idx = face.get("x_ancilla_idx")
        return None if idx is None else self._g(idx)

    def _append_cnot_pairs(self, pairs, *, circuit=None):
        cnot_targets = []
        for control, target in pairs:
            if control is not None and target is not None:
                cnot_targets.extend([control, target])
        if cnot_targets:
            (self.circuit if circuit is None else circuit).append(
                "CNOT",
                cnot_targets,
            )

    def _set_measurement_blocks(self, *blocks):
        self.measurement_blocks = tuple(blocks)
        self.circuit = stim.Circuit()
        for block in self.measurement_blocks:
            self.circuit += block


class ColorCodeSpaceMultiplexingBlock(_ColorCodeBlockBase):
    """
    Naive space-multiplexed color-code syndrome extraction block on superdense layout.

    Uses a 7-timeslice CNOT schedule with 6 CNOT positions per stabilizer type.
    """

    # Each entry: (z_neighbor_index_or_None, x_neighbor_index_or_None).
    # Indices refer to face["data_neighbors"].
    SCHEDULE = [
        (5, None),   # Timeslice 1: Z-pos5
        (0, 5),      # Timeslice 2: Z-pos0, X-pos5
        (1, 0),      # Timeslice 3: Z-pos1, X-pos0
        (4, 1),      # Timeslice 4: Z-pos4, X-pos1
        (3, 4),      # Timeslice 5: Z-pos3, X-pos4
        (2, 3),      # Timeslice 6: Z-pos2, X-pos3
        (None, 2),   # Timeslice 7: X-pos2
    ]

    def _build_circuit(self):
        self._require_layout("superdense")
        active_z_syn_indices = sorted(self.system.active_syndrome_indices_z)
        active_x_syn_indices = sorted(self.system.active_syndrome_indices_x)

        self.circuit.append("R", active_z_syn_indices)
        self.circuit.append("RX", active_x_syn_indices)
        self.circuit.append("TICK", tag="SE_start")

        for z_pos, x_pos in self.SCHEDULE:
            cnot_targets = []

            if z_pos is not None:
                for face in self._faces:
                    data_idx = self._z_neighbor(face, z_pos)
                    if data_idx is not None:
                        cnot_targets.extend([data_idx, self._z_ancilla(face)])

            if x_pos is not None:
                for face in self._faces:
                    data_idx = self._x_neighbor(face, x_pos)
                    if data_idx is not None:
                        cnot_targets.extend([self._x_ancilla(face), data_idx])

            if cnot_targets:
                self.circuit.append("CNOT", cnot_targets)
            self.circuit.append("TICK")

        self.circuit.append("M", active_z_syn_indices)
        self.circuit.append("MX", active_x_syn_indices)


class ColorCodeTimeMultiplexingBlock(_ColorCodeBlockBase):
    """Naive time-multiplexed block from Fig. 2: X check then Z check."""

    ORDER = [5, 0, 1, 4, 3, 2]

    def _build_circuit(self):
        self._require_layout("triangular")
        active_x_syn_indices = sorted(self.system.active_syndrome_indices_x)
        active_z_syn_indices = sorted(self.system.active_syndrome_indices_z)

        x_cycle = stim.Circuit()
        x_cycle.append("RX", active_x_syn_indices)
        x_cycle.append("TICK", tag="SE_start")
        for pos in self.ORDER:
            self._append_cnot_pairs(
                (
                    (self._x_ancilla(face), self._x_neighbor(face, pos))
                    for face in self._faces
                ),
                circuit=x_cycle,
            )
            x_cycle.append("TICK")
        x_cycle.append("MX", active_x_syn_indices)

        z_cycle = stim.Circuit()
        z_cycle.append("TICK")
        z_cycle.append("R", active_z_syn_indices)
        z_cycle.append("TICK", tag="SE_start")
        for pos in self.ORDER:
            self._append_cnot_pairs(
                (
                    (self._z_neighbor(face, pos), self._z_ancilla(face))
                    for face in self._faces
                ),
                circuit=z_cycle,
            )
            z_cycle.append("TICK")
        z_cycle.append("M", active_z_syn_indices)

        self._set_measurement_blocks(x_cycle, z_cycle)


class ColorCodeBellMultiplexingBlock(_ColorCodeBlockBase):
    """Bell-multiplexed ("superdense") color-code block from Fig. 2/3.

    This matches chromobius' ``superdense_color_code_X/Z`` round schedule:
    prepare an X/Z ancilla Bell pair, interact with three data directions in
    each CNOT orientation, then unprepare and measure the pair.
    """

    DATA_TO_ANCILLA_SCHEDULE = [(1, 2), (0, 3), (5, 4)]
    ANCILLA_TO_DATA_SCHEDULE = [(1, 2), (0, 3), (5, 4)]

    def _build_circuit(self):
        self._require_layout("superdense")
        active_x_syn_indices = sorted(self.system.active_syndrome_indices_x)
        active_z_syn_indices = sorted(self.system.active_syndrome_indices_z)

        self.circuit.append("RX", active_x_syn_indices)
        self.circuit.append("R", active_z_syn_indices)
        self.circuit.append("TICK", tag="SE_start")

        # Prepare one Bell pair per face across the two measurement ancillas.
        bell_pairs = [
            (self._x_ancilla(face), self._z_ancilla(face))
            for face in self._faces
        ]
        self._append_cnot_pairs(bell_pairs)
        self.circuit.append("TICK")

        for x_pos, z_pos in self.DATA_TO_ANCILLA_SCHEDULE:
            pairs = []
            for face in self._faces:
                pairs.append((self._x_neighbor(face, x_pos), self._x_ancilla(face)))
                pairs.append((self._z_neighbor(face, z_pos), self._z_ancilla(face)))
            self._append_cnot_pairs(pairs)
            self.circuit.append("TICK")

        for x_pos, z_pos in self.ANCILLA_TO_DATA_SCHEDULE:
            pairs = []
            for face in self._faces:
                pairs.append((self._x_ancilla(face), self._x_neighbor(face, x_pos)))
                pairs.append((self._z_ancilla(face), self._z_neighbor(face, z_pos)))
            self._append_cnot_pairs(pairs)
            self.circuit.append("TICK")

        # Bell-basis readout. Feedback is intentionally not inlined here.
        self._append_cnot_pairs(bell_pairs)
        self.circuit.append("TICK")
        self.circuit.append("MX", active_x_syn_indices)
        self.circuit.append("M", active_z_syn_indices)


class ColorCodeBellFlaggingBlock(_ColorCodeBlockBase):
    """Standard flag-style color-code block from Fig. 2.

    Each SE round combines an X-check measurement block and a Z-check
    measurement block. In each block one Bell ancilla accumulates the face
    stabilizer while the other produces a deterministic flag measurement.
    """

    DATA_SCHEDULE = [(0, 1), (2, 3), (4, 5)]

    def _build_circuit(self):
        self._require_layout("superdense")
        active_x_syn_indices = sorted(self.system.active_syndrome_indices_x)
        active_z_syn_indices = sorted(self.system.active_syndrome_indices_z)

        # X check with Z ancillas serving as flag/readout partners.
        x_cycle = stim.Circuit()
        x_cycle.append("R", active_z_syn_indices)
        x_cycle.append("RX", active_x_syn_indices)
        x_cycle.append("TICK", tag="SE_start")
        self._append_cnot_pairs(
            (
                (self._x_ancilla(face), self._z_ancilla(face))
                for face in self._faces
            ),
            circuit=x_cycle,
        )
        x_cycle.append("TICK")
        for z_side_pos, x_side_pos in self.DATA_SCHEDULE:
            pairs = []
            for face in self._faces:
                pairs.append((self._z_ancilla(face), self._z_neighbor(face, z_side_pos)))
                pairs.append((self._x_ancilla(face), self._x_neighbor(face, x_side_pos)))
            self._append_cnot_pairs(pairs, circuit=x_cycle)
            x_cycle.append("TICK")
        self._append_cnot_pairs(
            (
                (self._x_ancilla(face), self._z_ancilla(face))
                for face in self._faces
            ),
            circuit=x_cycle,
        )
        x_cycle.append("TICK")
        x_cycle.append("MX", active_x_syn_indices)
        x_cycle.append("M", active_z_syn_indices)

        # Z check with X ancillas serving as flag/readout partners.
        z_cycle = stim.Circuit()
        z_cycle.append("TICK")
        z_cycle.append("RX", active_x_syn_indices)
        z_cycle.append("R", active_z_syn_indices)
        z_cycle.append("TICK")
        self._append_cnot_pairs(
            (
                (self._x_ancilla(face), self._z_ancilla(face))
                for face in self._faces
            ),
            circuit=z_cycle,
        )
        z_cycle.append("TICK")
        for z_side_pos, x_side_pos in self.DATA_SCHEDULE:
            pairs = []
            for face in self._faces:
                pairs.append((self._z_neighbor(face, z_side_pos), self._z_ancilla(face)))
                pairs.append((self._x_neighbor(face, x_side_pos), self._x_ancilla(face)))
            self._append_cnot_pairs(pairs, circuit=z_cycle)
            z_cycle.append("TICK")
        self._append_cnot_pairs(
            (
                (self._x_ancilla(face), self._z_ancilla(face))
                for face in self._faces
            ),
            circuit=z_cycle,
        )
        z_cycle.append("TICK")
        z_cycle.append("MX", active_x_syn_indices)
        z_cycle.append("M", active_z_syn_indices)

        self._set_measurement_blocks(x_cycle, z_cycle)


class ColorCodeMiddleOutBlock(_ColorCodeBlockBase):
    """Inline-folding ("middle-out") circuit from Fig. 2/4.

    One SE round contains two alternating measurement layers.  Layer A and
    layer B each end in one contiguous MX/M readout layer; together they form
    the complete physical round.  Between the readouts, the CNOT schedule
    passes through the color-code frame and exchanges which representatives
    carry the X- and Z-basis measurements.
    """

    @staticmethod
    def memory_data_basis_map(patch, basis):
        plan = RectangleMiddleOutPlan.from_patch(patch)
        return plan.initial_basis_map(
            patch.data_indices,
            memory_basis=basis,
        )

    def _build_circuit(self):
        self._require_layout("rectangle")
        plan = RectangleMiddleOutPlan.from_patch(self._patch)
        layers = [
            [(self._g(a), self._g(b)) for a, b in layer]
            for layer in plan.cnot_layers
        ]
        x_measure = sorted(self._g(q) for q in plan.x_measure_indices)
        z_measure = sorted(self._g(q) for q in plan.z_measure_indices)
        self.data_qubits_initialized_by_block = set(x_measure) | set(z_measure)

        layer_a = stim.Circuit()
        layer_a.append("RX", z_measure)
        layer_a.append("R", x_measure)
        layer_a.append("TICK", tag="SE_start")
        for layer in layers:
            self._append_cnot_pairs(layer, circuit=layer_a)
            layer_a.append("TICK")

        layer_a.append("MX", x_measure)
        layer_a.append("M", z_measure)

        layer_b = stim.Circuit()
        layer_b.append("TICK")
        layer_b.append("RX", x_measure)
        layer_b.append("R", z_measure)
        layer_b.append("TICK")
        for layer in reversed(layers):
            self._append_cnot_pairs(layer, circuit=layer_b)
            layer_b.append("TICK")

        layer_b.append("MX", z_measure)
        layer_b.append("M", x_measure)

        self._set_measurement_blocks(layer_a, layer_b)


# Backwards-compatible name used by existing callers.
ColorCodeExtractionBlock = ColorCodeSpaceMultiplexingBlock
