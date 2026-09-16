"""Fixed-offset syndrome extraction for cyclic simplex-product patches."""

from typing import Any, Sequence

import stim

from .code_patch import SHYPSCode


class SHYPSCodeExtractionBlock:
    """Measure SHYPS gauges in three parallel CNOT layers per basis.

    For each offset in ``patch.gauge_offsets``, X ancilla ``(i, j)``
    interacts with data ``(i + offset, j)`` and Z ancilla ``(i, j)`` with
    data ``(i, j + offset)``, modulo the simplex length. Product indices
    stay in the patch's semantic frame under coordinate transformations.

    ``basis_order`` defaults to X then Z and allows single or repeated bases.
    Each basis has its own reset and terminal measurement block for the
    Builder/Tracker. This block inserts neither noise nor detectors. The
    cyclic offsets specify connectivity, not physical atom trajectories.
    """

    def __init__(self, system: Any, basis_order: Sequence[str] = ("X", "Z")):
        self.system = system
        self.basis_order = tuple(str(basis).upper() for basis in basis_order)
        if not self.basis_order or any(basis not in {"X", "Z"} for basis in self.basis_order):
            raise ValueError("basis_order must be a nonempty sequence of 'X'/'Z' bases.")

        gauges = {"X": [], "Z": []}
        for gauge in system.active_gauges:
            basis = gauge.get("type")
            support = set(gauge.get("data_indices", ()))
            if (basis not in gauges or len(support) != 3
                    or set(gauge.get("pauli", {})) != support
                    or set(gauge["pauli"].values()) != {basis}):
                raise ValueError("SHYPS extraction requires weight-3 pure-X or pure-Z gauges.")
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
        layers = [[], [], []]
        seen_ancillas = set()
        for gauge in sorted(gauges, key=lambda record: record.get("syn_idx") if record.get("syn_idx") is not None else -1):
            ancilla = gauge.get("syn_idx")
            if ancilla is None or ancilla not in self.system.syndrome_indices:
                raise ValueError("Every gauge requires a registered syndrome ancilla.")
            if ancilla in seen_ancillas:
                raise ValueError("Each gauge within one basis requires a distinct syndrome ancilla.")
            seen_ancillas.add(ancilla)

            patch_name = gauge.get("patch_name")
            owner = self.system.patches[patch_name][0]
            if not isinstance(owner, SHYPSCode):
                raise ValueError("SHYPS extraction requires gauges owned by SHYPSCode patches.")
            mapping = self.system.local_to_global_map[patch_name]
            pair = gauge.get("product_index")
            ancillas = owner.x_gauge_ancillas if basis == "X" else owner.z_gauge_ancillas
            if pair not in ancillas or mapping[ancillas[pair]] != ancilla:
                raise ValueError("SHYPS gauge product index does not match its ancilla.")
            i, j = pair
            m = owner.simplex_length
            targets = []
            for offset in owner.gauge_offsets:
                position = ((i + offset) % m, j) if basis == "X" else (i, (j + offset) % m)
                targets.append(mapping[owner.data_qubits[position]])
            if set(targets) != set(gauge["data_indices"]):
                raise ValueError("SHYPS gauge support does not match its declared cyclic offsets.")
            for layer, data in zip(layers, targets):
                layer.append((ancilla, data))

        for layer in layers:
            endpoints = [qubit for edge in layer for qubit in edge]
            if len(endpoints) != len(set(endpoints)):
                raise ValueError("SHYPS gauge interactions collide in a CNOT layer.")
        return layers
