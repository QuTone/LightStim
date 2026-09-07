"""Geometrically scheduled weight-2 Bacon-Shor gauge measurements."""

from typing import Any, Sequence

import stim


class BaconShorCodeExtractionBlock:
    """One noiseless SE round with a fixed schedule in each patch's frame.

    X ancillas interact with their left, then right data neighbors. Z ancillas
    interact with their negative-y, then positive-y data neighbors (top, then
    bottom in the row-index layout). Each basis uses two parallel CNOT layers.
    Patch rotations/transpositions transform these directions into global ones.

    ``basis_order`` defaults to X then Z; single or repeated bases are allowed.
    Each entry has its own reset and terminal measurement block so LightStim
    can propagate the noncommuting gauge measurements in their physical order.
    No detectors or noise are inserted here.
    """

    NEIGHBOR_OFFSETS = {
        "X": ((-1, 0), (+1, 0)),
        "Z": ((0, -1), (0, +1)),
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
            if (basis not in gauges or len(support) != 2
                    or set(gauge.get("pauli", {})) != support
                    or set(gauge["pauli"].values()) != {basis}):
                raise ValueError("Bacon-Shor extraction requires weight-2 pure-X or pure-Z gauges.")
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
                block.append("CX", targets)
                block.append("TICK")
            block.append("MX" if basis == "X" else "M", ancillas)
            blocks.append(block)

        self.measurement_blocks = tuple(blocks)
        self.circuit = sum(self.measurement_blocks, stim.Circuit())

    def _schedule_gauges(self, basis, gauges):
        if not gauges:
            return []
        layers = [[], []]
        seen_ancillas = set()
        for gauge in sorted(gauges, key=lambda gauge: gauge["syn_idx"] if gauge.get("syn_idx") is not None else -1):
            ancilla = gauge.get("syn_idx")
            if ancilla is None or ancilla not in self.system.syndrome_indices:
                raise ValueError("Every gauge requires a registered syndrome ancilla.")
            if ancilla in seen_ancillas:
                raise ValueError("Each gauge within one basis requires a distinct syndrome ancilla.")
            seen_ancillas.add(ancilla)
            owner = self.system.patches[gauge["patch_name"]][0]
            x, y = self.system.qubit_coords[ancilla]
            neighbors = []
            for offset in self.NEIGHBOR_OFFSETS[basis]:
                dx, dy = owner.transform_vector(offset)
                key = owner.get_grid_key((x + dx, y + dy))
                data = self.system.grid_map.get(key)
                if data not in gauge["data_indices"]:
                    raise ValueError(
                        f"Gauge ancilla {ancilla} does not have the declared Bacon-Shor {basis} neighbors."
                    )
                neighbors.append(data)
            if set(neighbors) != set(gauge["data_indices"]):
                raise ValueError(f"Gauge ancilla {ancilla} has inconsistent neighbor geometry.")
            for layer, data in zip(layers, neighbors):
                layer.append((ancilla, data))

        for layer in layers:
            endpoints = [qubit for edge in layer for qubit in edge]
            if len(endpoints) != len(set(endpoints)):
                raise ValueError("Bacon-Shor gauge interactions collide in a CNOT layer.")
        return layers
