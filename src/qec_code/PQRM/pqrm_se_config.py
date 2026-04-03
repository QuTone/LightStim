"""
PQRM SE schedule configuration: shared between normal and Hadamard-transformed versions.

6-tick bulk and boundary delta definitions (single source of truth).
"""

from typing import Dict, List, Tuple


# --- 6-tick schedule and boundary deltas (single source of truth) ---
TICK_DELTA_BULK: List[Tuple[int, int]] = [
    (-1, -1),   # Tick 0
    (+1, +1),   # Tick 1
    (+1, -1),   # Tick 2
    (-1, +1),   # Tick 3
    (0, 0),     # Tick 4
    (0, 0),     # Tick 5
]

TICK_DELTA_BOUNDARY: Dict[str, List[Tuple[int, int]]] = {
    "R1": [(-1, -3), (0, 0), (0, 0), (-1, -1), (-1, 1), (-1, 3)],
    "R2": [(-1, 5), (0, 0), (0, 0), (-1, 3), (-1, 1), (-1, -1)],
    "R3": [(-1, -7), (0, 0), (0, 0), (-1, -3), (-1, -1), (-1, 3)],
    "B1": [(3, -1), (0, 0), (1, -1), (0, 0), (-1, -1), (-3, -1)],
    "B2": [(-1, -1), (0, 0), (1, -1), (0, 0), (3, -1), (5, -1)],
    "B3": [(3, -1), (0, 0), (-1, -1), (0, 0), (-3, -1), (-7, -1)],
}


def get_boundary_data_deltas(region_key: str) -> List[Tuple[int, int]]:
    """Data qubit deltas (syn -> data) for a boundary region. Used by pqrm_patch."""
    tick_deltas = TICK_DELTA_BOUNDARY.get(region_key)
    if tick_deltas is None:
        return []
    return [(dx, dy) for dx, dy in tick_deltas if (dx, dy) != (0, 0)]
