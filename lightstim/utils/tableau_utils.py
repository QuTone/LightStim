"""
Tableau utilities: symplectic conversion for stabilizers.
"""

import numpy as np
from typing import List, Dict, Any, Set, Optional

def stabilizers_to_symplectic(
    system: Any,
    stabilizer_dicts: List[Dict[str, Any]],
    n: int,
) -> np.ndarray:
    """
    Convert stabilizer dicts to (k, 2n) symplectic matrix.

    Args:
        system: QECSystem with index_map, grid_map for coord->global resolution.
        stabilizer_dicts: List of stabilizer records with 'pauli' {key: "X"/"Z"/"Y"}.
                         Keys can be int (global index) or tuple (coord).
        n: Number of qubits (system.num_qubits).

    Returns:
        (k, 2n) uint8 matrix. Symplectic layout: X in cols 0..n-1, Z in cols n..2n-1.
    """
    rows = []
    index_map = getattr(system, "index_map", {})
    grid_map = getattr(system, "grid_map", {})

    def _to_global(key) -> Optional[int]:
        if isinstance(key, int):
            return key if 0 <= key < n else None
        if isinstance(key, tuple):
            snapped = (round(key[0], 6), round(key[1], 6)) if len(key) >= 2 else key
            return index_map.get(snapped) or index_map.get(key) or grid_map.get(snapped) or grid_map.get(key)
        return None

    for s in stabilizer_dicts:
        pauli = s.get("pauli", {})
        row = np.zeros(2 * n, dtype=np.uint8)
        for key, p in pauli.items():
            g = _to_global(key)
            if g is None and isinstance(key, int) and 0 <= key < n:
                g = key
            if g is None or g >= n:
                continue
            if p == "X":
                row[g] = 1
            elif p == "Z":
                row[n + g] = 1
            elif p == "Y":
                row[g] = 1
                row[n + g] = 1
        rows.append(row)

    if not rows:
        return np.zeros((0, 2 * n), dtype=np.uint8)
    return np.array(rows, dtype=np.uint8)
