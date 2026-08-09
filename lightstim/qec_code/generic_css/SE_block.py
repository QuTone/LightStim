"""Generic edge-coloration syndrome extraction for CSS codes.

This is intended as a conservative fallback for CSS code patches that provide
stabilizer supports but do not yet have a code-specific syndrome-extraction
schedule. X and Z checks are measured with independent bipartite edge colorings.
"""

from typing import Any, Dict, List, Sequence, Tuple

import stim

from .bipartite_edge_coloring import color_bipartite_edges


EdgePayload = Tuple[int, int]


class GenericCSSColorationExtractionBlock:
    """
    Syndrome extraction by bipartite edge coloring.

    For each CSS basis, the Tanner graph between syndrome ancillas and data
    qubits is edge-colored. Every color is one CNOT layer, so a basis with max
    Tanner degree Delta uses Delta CNOT layers. The default basis order is X
    then Z, giving depth ``depth_x + depth_z``.
    """

    def __init__(self, system: Any, basis_order: Sequence[str] = ("X", "Z")):
        self.system = system
        self.basis_order = tuple(b.upper() for b in basis_order)
        if set(self.basis_order) != {"X", "Z"} or len(self.basis_order) != 2:
            raise ValueError("basis_order must be a permutation of ('X', 'Z').")

        self.circuit = stim.Circuit()
        self.x_layers = self._color_stabilizers(self.system.active_stabilizers_x)
        self.z_layers = self._color_stabilizers(self.system.active_stabilizers_z)
        self.depth_x = len(self.x_layers)
        self.depth_z = len(self.z_layers)
        self.cnot_depth = self.depth_x + self.depth_z
        self._build_circuit()

    def _build_circuit(self):
        active_syn_indices = sorted(self.system.active_syndrome_indices)
        active_x_syn_indices = sorted(self.system.active_syndrome_indices_x)

        self.circuit.append("R", active_syn_indices)
        self.circuit.append("TICK", tag="SE_start")

        if active_x_syn_indices:
            self.circuit.append("H", active_x_syn_indices)
        self.circuit.append("TICK")

        for basis in self.basis_order:
            layers = self.x_layers if basis == "X" else self.z_layers
            for layer in layers:
                cnot_targets: List[int] = []
                for syn_idx, data_idx in layer:
                    if basis == "X":
                        cnot_targets.extend([syn_idx, data_idx])
                    else:
                        cnot_targets.extend([data_idx, syn_idx])
                if cnot_targets:
                    self.circuit.append("CNOT", cnot_targets)
                self.circuit.append("TICK")

        if active_x_syn_indices:
            self.circuit.append("H", active_x_syn_indices)
        self.circuit.append("TICK")

        self.circuit.append("M", active_syn_indices)

    @staticmethod
    def _color_stabilizers(stabilizers: Sequence[Dict[str, Any]]) -> List[List[EdgePayload]]:
        edges: List[EdgePayload] = []
        for stab in stabilizers:
            syn_idx = stab.get("syn_idx")
            if syn_idx is None:
                raise ValueError("Generic CSS extraction requires every stabilizer to have a syndrome qubit.")
            for data_idx in stab.get("data_indices", []):
                edges.append((syn_idx, data_idx))

        return color_bipartite_edges(edges)
