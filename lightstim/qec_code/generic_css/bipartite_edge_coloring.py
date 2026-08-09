"""Deterministic optimal edge coloring for bipartite Tanner graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Hashable, Optional, Sequence, TypeVar


LeftNode = TypeVar("LeftNode", bound=Hashable)
RightNode = TypeVar("RightNode", bound=Hashable)
Edge = tuple[LeftNode, RightNode]


def color_bipartite_edges(edges: Sequence[Edge]) -> list[list[Edge]]:
    """Partition bipartite edges into a minimum number of matchings.

    The two entries of an edge live in separate vertex namespaces.  Returned
    colors are deterministic for a fixed input order and their count equals the
    maximum Tanner degree, by Konig's line-coloring theorem.

    Args:
        edges: ``(left_node, right_node)`` pairs. Parallel edges are allowed.

    Returns:
        A list of colors, each represented by a conflict-free list of edges.
    """
    if not edges:
        return []

    left_values = list(dict.fromkeys(left for left, _ in edges))
    right_values = list(dict.fromkeys(right for _, right in edges))
    left_index = {value: index for index, value in enumerate(left_values)}
    right_index = {value: index for index, value in enumerate(right_values)}

    n_left_real = len(left_values)
    n_right_real = len(right_values)
    degree_left = [0] * n_left_real
    degree_right = [0] * n_right_real

    buckets: DefaultDict[tuple[int, int], list[Optional[Edge]]] = defaultdict(list)
    for edge in edges:
        left = left_index[edge[0]]
        right = right_index[edge[1]]
        degree_left[left] += 1
        degree_right[right] += 1
        buckets[(left, right)].append(edge)

    delta = max(max(degree_left, default=0), max(degree_right, default=0))
    if delta == 0:
        return []

    # Regularize to a balanced Delta-regular bipartite multigraph. Repeated
    # perfect matchings then give an optimal Delta-edge-coloring.
    n_vertices = max(n_left_real, n_right_real)
    degree_left.extend([0] * (n_vertices - n_left_real))
    degree_right.extend([0] * (n_vertices - n_right_real))

    left_deficits: list[int] = []
    right_deficits: list[int] = []
    for left, degree in enumerate(degree_left):
        left_deficits.extend([left] * (delta - degree))
    for right, degree in enumerate(degree_right):
        right_deficits.extend([right] * (delta - degree))

    if len(left_deficits) != len(right_deficits):
        raise RuntimeError(
            "Internal edge-coloring error: unbalanced regularization deficits."
        )

    for left, right in zip(left_deficits, right_deficits):
        buckets[(left, right)].append(None)

    colors: list[list[Edge]] = []
    for _ in range(delta):
        adjacency = [[] for _ in range(n_vertices)]
        for (left, right), payloads in buckets.items():
            if payloads:
                adjacency[left].append(right)
        for neighbors in adjacency:
            neighbors.sort()

        matching = _hopcroft_karp_perfect(adjacency, n_vertices, n_vertices)
        color: list[Edge] = []
        for left, right in enumerate(matching):
            payloads = buckets[(left, right)]
            payload = payloads.pop()
            if not payloads:
                del buckets[(left, right)]
            if payload is not None:
                color.append(payload)

        color.sort(key=lambda edge: (left_index[edge[0]], right_index[edge[1]]))
        colors.append(color)

    if buckets:
        raise RuntimeError("Internal edge-coloring error: uncolored edges remain.")
    return colors


def _hopcroft_karp_perfect(
    adjacency: Sequence[Sequence[int]],
    n_left: int,
    n_right: int,
) -> list[int]:
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
        raise RuntimeError(
            "Internal edge-coloring error: regularized graph has no perfect matching."
        )
    return pair_left


__all__ = ["color_bipartite_edges"]
