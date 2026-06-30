"""Generic edge-coloration syndrome extraction for CSS codes.

This is intended as a conservative fallback for CSS code patches that provide
stabilizer supports but do not yet have a code-specific syndrome-extraction
schedule. X and Z checks are measured with independent bipartite edge colorings.
"""

from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import stim


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

        return _edge_color_bipartite(edges)


def _edge_color_bipartite(edges: Sequence[EdgePayload]) -> List[List[EdgePayload]]:
    """Optimal edge coloring for a bipartite graph using repeated matchings."""
    if not edges:
        return []

    left_values = sorted({syn_idx for syn_idx, _ in edges})
    right_values = sorted({data_idx for _, data_idx in edges})
    left_index = {value: idx for idx, value in enumerate(left_values)}
    right_index = {value: idx for idx, value in enumerate(right_values)}

    n_left_real = len(left_values)
    n_right_real = len(right_values)
    degree_left = [0] * n_left_real
    degree_right = [0] * n_right_real

    buckets: DefaultDict[Tuple[int, int], List[Optional[EdgePayload]]] = defaultdict(list)
    for edge in edges:
        left = left_index[edge[0]]
        right = right_index[edge[1]]
        degree_left[left] += 1
        degree_right[right] += 1
        buckets[(left, right)].append(edge)

    delta = max(max(degree_left, default=0), max(degree_right, default=0))
    if delta == 0:
        return []

    n_vertices = max(n_left_real, n_right_real)
    degree_left.extend([0] * (n_vertices - n_left_real))
    degree_right.extend([0] * (n_vertices - n_right_real))

    left_deficits: List[int] = []
    right_deficits: List[int] = []
    for left, degree in enumerate(degree_left):
        left_deficits.extend([left] * (delta - degree))
    for right, degree in enumerate(degree_right):
        right_deficits.extend([right] * (delta - degree))

    if len(left_deficits) != len(right_deficits):
        raise RuntimeError("Internal edge-coloring error: unbalanced regularization deficits.")

    for left, right in zip(left_deficits, right_deficits):
        buckets[(left, right)].append(None)

    layers: List[List[EdgePayload]] = []
    for _ in range(delta):
        adjacency = [[] for _ in range(n_vertices)]
        for (left, right), payloads in buckets.items():
            if payloads:
                adjacency[left].append(right)
        for neighbors in adjacency:
            neighbors.sort()

        matching = _hopcroft_karp_perfect(adjacency, n_vertices, n_vertices)
        layer: List[EdgePayload] = []

        for left, right in enumerate(matching):
            payloads = buckets[(left, right)]
            payload = payloads.pop()
            if not payloads:
                del buckets[(left, right)]
            if payload is not None:
                layer.append(payload)

        layer.sort()
        layers.append(layer)

    if buckets:
        raise RuntimeError("Internal edge-coloring error: uncolored edges remain.")

    return layers


def _hopcroft_karp_perfect(
    adjacency: Sequence[Sequence[int]],
    n_left: int,
    n_right: int,
) -> List[int]:
    pair_left = [-1] * n_left
    pair_right = [-1] * n_right
    dist = [0] * n_left

    def bfs() -> bool:
        queue: deque[int] = deque()
        found_free = False
        for left in range(n_left):
            if pair_left[left] == -1:
                dist[left] = 0
                queue.append(left)
            else:
                dist[left] = -1

        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                matched_left = pair_right[right]
                if matched_left == -1:
                    found_free = True
                elif dist[matched_left] == -1:
                    dist[matched_left] = dist[left] + 1
                    queue.append(matched_left)
        return found_free

    def dfs(left: int) -> bool:
        for right in adjacency[left]:
            matched_left = pair_right[right]
            if matched_left == -1 or (
                dist[matched_left] == dist[left] + 1 and dfs(matched_left)
            ):
                pair_left[left] = right
                pair_right[right] = left
                return True
        dist[left] = -1
        return False

    matching_size = 0
    while bfs():
        for left in range(n_left):
            if pair_left[left] == -1 and dfs(left):
                matching_size += 1

    if matching_size != n_left:
        raise RuntimeError("Internal edge-coloring error: regularized graph has no perfect matching.")
    return pair_left
