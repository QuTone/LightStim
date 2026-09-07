"""Fixed local X-then-Z gauge extraction for the planar subsystem surface code."""

from typing import Any, Sequence

import stim


class SubsystemSurfaceCodeExtractionBlock:
    """One noiseless SE cycle, with a separate physical block per gauge basis.

    Each basis uses the four directional slots of the rotated surface code's
    perpendicular schedule, skipping absent or out-of-support neighbors. Bulk
    ancillas execute three CNOTs and boundary ancillas execute two. The default
    X-then-Z cycle has eight CNOT layers, plus preparation and readout; this is
    a simple baseline, not the paper's four-step interleaved schedule.

    ``basis_order`` also accepts single or repeated X/Z phases. All supports
    come from declared gauges; center declarations remain fixed. LightStim's
    Builder/Tracker derives detectors from these physical measurement blocks.
    """

    NEIGHBOR_OFFSETS = {
        "X": ((+1, +1), (-1, +1), (+1, -1), (-1, -1)),
        "Z": ((+1, +1), (+1, -1), (-1, +1), (-1, -1)),
    }

    def __init__(self, system: Any, basis_order: Sequence[str] = ("X", "Z")):
        self.system = system
        self.basis_order = tuple(str(basis).upper() for basis in basis_order)
        if not self.basis_order or any(basis not in {"X", "Z"} for basis in self.basis_order):
            raise ValueError("basis_order must be a nonempty sequence of 'X'/'Z' bases.")

        gauges = {"X": [], "Z": []}
        for gauge in system.active_gauges:
            basis = gauge.get("type")
            support = set(gauge.get("data_indices", ()))
            if (basis not in gauges or len(support) not in {2, 3}
                    or set(gauge.get("pauli", {})) != support
                    or set(gauge["pauli"].values()) != {basis}):
                raise ValueError("Subsystem surface extraction requires weight-2/3 pure-X or pure-Z gauges.")
            if not support <= system.data_indices:
                raise ValueError("Gauge supports must contain registered data qubits only.")
            gauges[basis].append(gauge)

        layers = {basis: self._schedule_gauges(basis, records) for basis, records in gauges.items()}
        self.x_layers, self.z_layers = layers["X"], layers["Z"]
        self.depth_x, self.depth_z = len(self.x_layers), len(self.z_layers)
        self.cnot_depth = sum(len(layers[basis]) for basis in self.basis_order)

        blocks = []
        for basis in self.basis_order:
            ancillas = sorted(gauge["syn_idx"] for gauge in gauges[basis])
            if not ancillas:
                raise ValueError(f"No active {basis} gauges are available for the requested measurement block.")
            block = stim.Circuit()
            if blocks:
                block.append("TICK")
            block.append("RX" if basis == "X" else "R", ancillas)
            block.append("TICK", tag="SE_start")
            for layer in layers[basis]:
                targets = []
                for ancilla, data in layer:
                    targets.extend((ancilla, data) if basis == "X" else (data, ancilla))
                if targets:
                    block.append("CX", targets)
                block.append("TICK")
            block.append("MX" if basis == "X" else "M", ancillas)
            blocks.append(block)

        self.measurement_blocks = tuple(blocks)
        self.circuit = sum(self.measurement_blocks, stim.Circuit())

    def _schedule_gauges(self, basis, gauges):
        if not gauges:
            return []
        layers = [[], [], [], []]
        seen_ancillas = set()
        for gauge in sorted(gauges, key=lambda g: g["syn_idx"] if g.get("syn_idx") is not None else -1):
            ancilla = gauge.get("syn_idx")
            if ancilla is None or ancilla not in self.system.syndrome_indices:
                raise ValueError("Every gauge requires a registered syndrome ancilla.")
            if ancilla in seen_ancillas:
                raise ValueError("Each gauge within one basis requires a distinct syndrome ancilla.")
            seen_ancillas.add(ancilla)
            owner = self.system.patches[gauge["patch_name"]][0]
            x, y = self.system.qubit_coords[ancilla]
            neighbors = set()
            for layer, offset in zip(layers, self.NEIGHBOR_OFFSETS[basis]):
                dx, dy = owner.transform_vector(offset)
                key = owner.get_grid_key((x + dx, y + dy))
                data = self.system.grid_map.get(key)
                if data in gauge["data_indices"]:
                    layer.append((ancilla, data))
                    neighbors.add(data)
            if neighbors != set(gauge["data_indices"]):
                raise ValueError(f"Gauge ancilla {ancilla} has support outside its diagonal neighbors.")

        for layer in layers:
            endpoints = [q for edge in layer for q in edge]
            if len(endpoints) != len(set(endpoints)):
                raise ValueError("Subsystem surface interactions collide in a CNOT layer.")
        return layers
