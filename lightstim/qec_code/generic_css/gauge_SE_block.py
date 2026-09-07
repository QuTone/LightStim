"""Physical CSS gauge measurements with explicit measurement-block boundaries."""

from typing import Any, Sequence

import stim

from .bipartite_edge_coloring import color_bipartite_edges


class GenericCSSGaugeExtractionBlock:
    """Measure CSS gauges one commuting basis group at a time.

    ``basis_order`` is any nonempty sequence of X/Z bases, including a single
    basis or repeated bases. Each entry produces its own reset, Clifford, and
    terminal-readout block. Noncommuting X/Z gauges are never combined into a
    single measurement block. Same-basis CNOT layers use bipartite edge coloring;
    this generic schedule does not promise a particular circuit-level distance.

    Each gauge must have a distinct ancilla within its basis. Ancillas may be
    reused across different bases because each block explicitly resets them.
    Algebraically redundant gauge generators are allowed.
    """

    def __init__(self, system: Any, basis_order: Sequence[str] = ("X", "Z")):
        self.system = system
        self.basis_order = tuple(str(basis).upper() for basis in basis_order)
        if not self.basis_order or any(basis not in {"X", "Z"} for basis in self.basis_order):
            raise ValueError("basis_order must be a nonempty sequence of 'X'/'Z' bases.")

        by_basis = {"X": [], "Z": []}
        for gauge in system.active_gauges:
            basis = gauge.get("type")
            if basis not in by_basis or set(gauge.get("pauli", {}).values()) != {basis}:
                raise ValueError("Generic CSS gauge extraction requires nonempty pure-X or pure-Z gauges.")
            by_basis[basis].append(gauge)
        if not any(by_basis.values()):
            raise ValueError("Generic CSS gauge extraction requires active gauge generators.")

        ancillas = {}
        layers = {}
        for basis, gauges in by_basis.items():
            seen = set()
            edges = []
            for gauge in gauges:
                ancilla = gauge.get("syn_idx")
                if ancilla is None or ancilla not in system.syndrome_indices:
                    raise ValueError("Every gauge requires a registered syndrome ancilla.")
                if ancilla in seen:
                    raise ValueError("Each gauge within one basis requires a distinct syndrome ancilla.")
                seen.add(ancilla)
                support = gauge.get("data_indices", ())
                if not set(support) <= system.data_indices:
                    raise ValueError("Gauge supports must contain registered data qubits only.")
                edges.extend((ancilla, data) for data in support)
            ancillas[basis] = sorted(seen)
            layers[basis] = color_bipartite_edges(edges)

        self.x_layers = layers["X"]
        self.z_layers = layers["Z"]
        self.depth_x = len(self.x_layers)
        self.depth_z = len(self.z_layers)
        self.cnot_depth = sum(len(layers[basis]) for basis in self.basis_order)
        blocks = []
        for basis in self.basis_order:
            if not ancillas[basis]:
                raise ValueError(f"No active {basis} gauges are available for the requested measurement block.")
            block = stim.Circuit()
            if blocks:
                block.append("TICK")
            block.append("RX" if basis == "X" else "R", ancillas[basis])
            block.append("TICK", tag="SE_start")
            for layer in layers[basis]:
                targets = []
                for ancilla, data in layer:
                    targets.extend((ancilla, data) if basis == "X" else (data, ancilla))
                block.append("CX", targets)
                block.append("TICK")
            block.append("MX" if basis == "X" else "M", ancillas[basis])
            blocks.append(block)

        self.measurement_blocks = tuple(blocks)
        self.circuit = stim.Circuit()
        for block in self.measurement_blocks:
            self.circuit += block
